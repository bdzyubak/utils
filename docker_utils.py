import pandas as pd
import json
import requests


def add_gcc_to_dockerfile(model_dir):
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


def post_json_get_preds(X_json):
    headers = {"content-type": "application/json", "Accept": "text/plain"}
    response = requests.post("http://127.0.0.1:8000/invocations", data=X_json, headers=headers)
    preds_dict = json.loads(response.content)
    if response.status_code == 200:
        print('Successfully predicted using Docker container!')
    else:
        print(f'Error predicting using docker: {response.content}')
    return preds_dict['predictions']
