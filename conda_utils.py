import os
from pathlib import Path
from typing import Optional, Union

from os_utils import run_command


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


def develop_submodules(env_name: str):
    # This is needed to run in command line; for Pycharm use Settings->Project->Project Structure and add the below
    # submodules as source folders
    path_source = Path(__file__).parents[1]
    for submodule in ['nnUnet', 'utils']:
        submodule_path = str(path_source / submodule)
        command_add_submodule_path = 'conda develop ' + submodule_path + ' -n ' + env_name
        retcode, text = run_command(command_add_submodule_path)
        print(f'Developed submodule path {submodule_path} to {env_name}')


def conda_create_from_yml(env_name: str, file: Union[str, Path]):
    command_conda_create = f'conda env create -n {env_name} -f {file}'

    retcode, text = run_command(command_conda_create)
    if retcode == 0:
        print(f'Conda environment {env_name} created successfully.')
    else:
        print(f'Conda environment {env_name} failed to be created. Check write permissions in conda install dir.')
        print(text)


def conda_extend_env(env_name: str, file: Union[str, Path]):
    command_conda_update = f'conda env update --name {env_name} --file {file} --prune'

    retcode, text = run_command(command_conda_update)
    if retcode == 0:
        print(f'Conda environment {env_name} created successfully.')
    else:
        print(f'Conda environment {env_name} failed to be created. Check write permissions in conda install dir.')
        print(text)


def conda_create(env_name: str):
    command_conda_create = 'conda create -n ' + env_name + ' -y'

    retcode, text = run_command(command_conda_create)
    if retcode == 0:
        print(f'Conda environment {env_name} created successfully.')
    else:
        print(f'Conda environment {env_name} failed to be created. Check write permissions in conda install dir.')
        print(text)


def install_hard_linked_pytorch(env_name: str):
    # Install dependencies via hard-link to conserve space
    command_install = (f'conda run -n {env_name} conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c '
                       f'pytorch -c nvidia')
    run_command(command_install)


def install_shared_dependencies(env_name: str, shared_deps: list[str], verbose: Optional[bool] = None):
    # Install dependencies that are not part of submodule requirements but are useful for running torch-control
    # experiments without switching to a dedicated interpreter.  Using pip to install since some deps are not on the
    # conda default channel
    print(f'Installing shared deps to {env_name}.')
    install_failed = dict()
    for shared_dep in shared_deps:
        # Use conda to hard link and conserve space
        command_install = f'conda install -n {env_name} {shared_dep} -y'
        retcode, text = run_command(command_install, verbose=verbose)

        if retcode != 0:
            # Conda install failed, fall back to pip inside conda env
            # NB: If you did not do conda install -n [env_name] pip, deps will install to base!
            command_install = f'conda run -n {env_name} python -m pip install {shared_dep}'
            retcode, text = run_command(command_install, verbose=verbose)
            if retcode != 0:
                install_failed[shared_dep] = text
    if install_failed:
        raise OSError(f"Failed to install dependencies {install_failed}")
