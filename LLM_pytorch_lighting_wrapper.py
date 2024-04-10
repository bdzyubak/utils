from lightning.pytorch import loggers
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score

import torch
from torch.optim import AdamW
import lightning as pl
import mlflow

from transformers import (DistilBertTokenizer, DistilBertForSequenceClassification, BertForSequenceClassification,
                          BertTokenizer, RobertaTokenizer, RobertaModel, AutoModelForCausalLM, AutoTokenizer)

from torch_utils import freeze_layers, get_model_size
from utils.torch_utils import tensor_to_numpy, average_round_metric

# NB: Speed up processing for negligible loss of accuracy. Verify acceptable accuracy for a production use case
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.allow_tf32 = True

supported_models = {'distilbert': 'distilbert-base-uncased', 'bert': 'bert-base-uncased',
                    'roberta': 'FacebookAI/roberta-base', 'llama': 'TheBloke/llama-2-70b-Guanaco-QLoRA-fp16'}


class FineTuneLLM(pl.LightningModule):
    def __init__(self, model_name, num_classes, device='cuda:0', learning_rate=5e-5, do_layer_freeze=True):
        super(FineTuneLLM, self).__init__()
        self.set_up_model_and_tokenizer(device, do_layer_freeze, model_name, num_classes)

        self.learning_rate = learning_rate

    def set_up_model_and_tokenizer(self, device, do_layer_freeze, model_name, num_classes):
        check_model_supported(model_name)
        # TODO: explore swapping tokenizers. For now, use native
        if model_name == 'distilbert':
            model = DistilBertForSequenceClassification.from_pretrained(supported_models['distilbert'],
                                                                        num_labels=num_classes)
            self.fine_tune_head = ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
        elif model_name == 'bert':
            model = BertForSequenceClassification.from_pretrained(supported_models['bert'], num_labels=num_classes)
            self.fine_tune_head = ['classifier.bias', 'classifier.weight']
        elif model_name == 'roberta':
            raise NotImplementedError(f"Trainer needs to pass arguments differently. labels keyword not found.")
            model = RobertaModel.from_pretrained(supported_models['roberta'],
                                                 num_labels=num_classes)
            self.fine_tune_head = ['pooler.dense.bias', 'pooler.dense.weight']
        elif model_name.lower() == 'llama':
            raise NotImplementedError(f"Not tested.")
            model = AutoModelForCausalLM.from_pretrained(supported_models['llama'],
                                                         num_classes=num_classes)
        else:
            raise NotImplementedError(f"Support for the model {model_name} has not been implemented.")

        self.model = model
        if do_layer_freeze:
            self.model = freeze_layers(self.fine_tune_head, self.model)
            mlflow.log_param("Freeze layers except", self.fine_tune_head)
        self.model.to(device)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def on_train_epoch_start(self):
        self.train_loss = list()
        self.train_acc = list()

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        labels_class = tensor_to_numpy(batch['labels_class'])
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        preds_batch = tensor_to_numpy(torch.argmax(outputs['logits'], axis=1))
        train_acc_batch = accuracy_score(preds_batch, labels_class)
        self.train_loss.append(tensor_to_numpy(loss))
        self.train_acc.append(train_acc_batch)
        return loss

    def on_train_epoch_end(self):
        self.log('train_loss', average_round_metric(self.train_loss))
        self.log('train_acc', average_round_metric(self.train_acc))

    def on_validation_epoch_start(self):
        self.val_loss = list()
        self.val_acc = list()

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        labels_class = tensor_to_numpy(batch['labels_class'])
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        preds_batch = tensor_to_numpy(torch.argmax(outputs['logits'], axis=1))
        val_acc_batch = accuracy_score(preds_batch, labels_class)
        loss = outputs.loss
        self.val_loss.append(tensor_to_numpy(loss))
        self.val_acc.append(val_acc_batch)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log('val_loss', average_round_metric(self.val_loss))
        self.log('val_acc', average_round_metric(self.val_acc))

    def configure_optimizers(self):
        self.optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return self.optimizer


def check_model_supported(model_name):
    if model_name not in supported_models:
        raise NotImplementedError(f"The model support for {model_name} has not been implemented.")


def tokenizer_setup(tokenizer_name):
    check_model_supported(tokenizer_name)
    if tokenizer_name == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained(supported_models['distilbert'])
    elif tokenizer_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained(supported_models['bert'])
    elif tokenizer_name.startswith('roberta'):
        tokenizer = RobertaTokenizer.from_pretrained(supported_models['roberta'])
    elif tokenizer_name.startswith('llama'):
        tokenizer = AutoTokenizer.from_pretrained(supported_models['llama'])
    else:
        raise NotImplementedError()
    return tokenizer


# def qc_requested_models_supported(model_names):
#     models_unsupported = list()
#     for model_name in model_names:
#         try:
#             model = FineTuneLLM(num_classes=1, model_name=model_name)
#         except RuntimeError:
#             models_unsupported.append(model_name)
#     if models_unsupported:
#         raise ValueError(f'The following models are not supported {models_unsupported}')


def model_setup(save_dir, num_classes, model_name='distilbert-base-uncased', do_layer_freeze=True):
    model_name_clean = model_name.split('\\')[-1]
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir,
                                          filename=model_name_clean + "-{epoch:02d}-{val_loss:.2f}",
                                          save_top_k=1,
                                          monitor="val_acc")
    early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.0001, patience=5, verbose=False, mode="max")
    tb_logger = loggers.TensorBoardLogger(save_dir=save_dir)
    model = FineTuneLLM(model_name=model_name, num_classes=num_classes,
                        do_layer_freeze=do_layer_freeze)

    trainer = pl.Trainer(max_epochs=100, callbacks=[checkpoint_callback, early_stop_callback], logger=tb_logger,
                         log_every_n_steps=50)
    return model, trainer
