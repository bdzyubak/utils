import pandas as pd
from pathlib import Path
from typing import Union
import json
import requests

from os_utils import str_to_path


def add_gcc_to_dockerfile(model_dir: Union[Path, str]):
    model_dir = str_to_path(model_dir)
    print(f"DEPRECATION WARNING: Dockerfiles are now stored in \\dockerfiles. Use those to build rather than modifying "
          f"on the fly. ")
    with open(model_dir / 'Dockerfile', 'r') as f:
        lines = f.readlines()

    for ind, line in enumerate(lines):
        if line.startswith('RUN apt-get -y update && apt-get install -y --no-install-recommends '):
            lines[ind] = line.split('\n')[0] + ' -y gcc g++\n'

    with open(model_dir / 'Dockerfile', 'w') as f:
        f.writelines(lines)


def convert_dataframe_to_json_for_docker(X: pd.DataFrame) -> dict:
    # Convert dataframe into a format accepted by mlflow served model
    # Currently, integration tested via projects/MachineLearning/energy_use_time_series_forecasting/build_inference_docker_container.py
    dict_json = {'inputs': dict()}
    for feature in X.columns:
        dict_json['inputs'][feature] = X[feature].values.astype(int).tolist()
    X_json = json.dumps(dict_json)
    return X_json


def post_json_get_preds(X_json: dict, port: Union[str, int]='8000'):
    headers = {"content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(f"http://127.0.0.1:{port}/invocations", data=X_json, headers=headers)
    preds_dict = json.loads(response.content)
    if response.status_code == 200:
        print('Successfully predicted using Docker container!')
    else:
        print(f'Error predicting using docker: {response.content}')
    return preds_dict['predictions']
