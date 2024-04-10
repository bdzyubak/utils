import pandas as pd
from pathlib import Path
from typing import Optional, Union

import seaborn as sns
import tbparse
from matplotlib import pyplot as plt

from utils.os_utils import filename_to_title


def set_plotting_defaults():
    # Import this to set defaults, no need to run
    plt.rcParams["figure.figsize"] = (9.6, 7.2)  # must set at the top
    plt.rcParams.update({'font.size': 22})  # must set at the top
    sns.set()


# Runs when anything is imported from this module. Import set_plotting_defaults() if you will just be plotting in the
# script
set_plotting_defaults()


def lineplot(df: pd.DataFrame, save_file: Optional[Path] = None, xlabel: Optional[str] = None,
             ylabel: Optional[str] = None, legend_outside_plot: bool = True):
    plt.figure()
    color_palette = sns.color_palette()
    # Skip first column which is the index
    plot = sns.lineplot(data=df[df.columns[1:]], palette=color_palette, legend=True)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    title = filename_to_title(save_file)
    plt.title(title)
    if legend_outside_plot:
        sns.move_legend(plot.axes, 'center right')
    else:
        sns.move_legend(plot.axes, 'center right', bbox_to_anchor=(1, 0.5), ncol=1)

    plt.savefig(save_file)
    plt.show()


def plot_tensorboard_logs(model_dir, model_version, title=None, log_type='lightning_logs'):
    reader = tbparse.SummaryReader(str(model_dir / log_type / model_version))
    df = reader.scalars
    # Deal with epoch being logged twice on train end and val end
    epochs = df[df['tag'] == 'epoch'].iloc[0::2, :]
    metrics = ['loss', 'acc']
    fig, ax = plt.subplots(1, len(metrics), figsize=(19.2, 7.2))
    if title is not None:
        fig.suptitle(title)
    for ax_idx, metric in enumerate(metrics):
        metric_values = df[df['tag'].isin(['train_' + metric, 'val_' + metric])]
        metric_values.loc[metric_values['tag'] == 'train_' + metric, 'epoch'] = list(epochs['value'])
        metric_values.loc[metric_values['tag'] == 'val_' + metric, 'epoch'] = list(epochs['value'])
        # fig, ax = plt.subplots(1, len(metrics))
        ax[ax_idx].set_title('Metric: ' + metric.capitalize())
        sns.lineplot(ax=ax[ax_idx], x='epoch', y='value', hue='tag', data=metric_values)
        # Drop the hue label name
        handles, labels = ax[ax_idx].get_legend_handles_labels()
        ax[ax_idx].legend(handles=handles[:], labels=labels[:])
        ax[ax_idx].set(ylabel=metric)
    plt.savefig(model_dir / f"training_metrics_{model_version}.png", bbox_inches="tight")
    plt.show()
