import os
from pathlib import Path
from typing import Optional

from os_utilities import run_command


def check_conda_installed():
    if os.name == 'nt':
        check_conda_command = 'where conda'
    else:
        check_conda_command = 'which conda'
    print('Conda found at: ')
    retcode, text = run_command(check_conda_command)
    if retcode != 0:
        raise OSError('Conda installation not found.')


def execute_setup_py_in_conda_env(env_name: str, submodule_name: Optional[str] = None):
    if submodule_name is None:
        submodule_name = env_name

    path_submodule_setup = "pip install -e " + (os.path.join(Path(__file__).parents[1] / submodule_name, '.'))
    command_conda_setup = 'conda run -n ' + env_name + ' ' + path_submodule_setup
    retcode, text = run_command(command_conda_setup)
    if retcode == 0:
        print(f'Dependencies installed successfully.')
    else:
        print(f'Failed to install dependencies. {text}')


def develop_submodules(env_name: str, develop_paths: bool = True):
    # This is needed to run in command line; for Pycharm use Settings->Project->Project Structure and add the below
    # submodules as source folders
    path_source = Path(__file__).parents[1]
    for submodule in ['nnUnet', 'utils']:
        submodule_path = str(path_source / submodule)
        command_add_submodule_path = 'conda develop ' + submodule_path + ' -n ' + env_name
        retcode, text = run_command(command_add_submodule_path)
        print(f'Developed submodule path {submodule_path} to {env_name}')


def conda_create(env_name) -> str:
    command_conda_create = 'conda create -n ' + env_name + ' -y'

    retcode, text = run_command(command_conda_create)
    if retcode == 0:
        print(f'Conda environment {env_name} created successfully.')
    else:
        print(f'Conda environment {env_name} failed to be created. Check write permissions in conda install dir.')
        print(text)
    return env_name


def install_hard_linked_pytorch(env_name):
    # Install dependencies via hard-link to conserve space
    command_install = (f'conda run -n {env_name} conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c '
                       f'pytorch -c nvidia')
    run_command(command_install)
