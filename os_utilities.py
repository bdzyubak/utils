import os
import pathlib
import subprocess
from typing import Union


def run_command(command: Union[str, list]):
    output = subprocess.run(command, capture_output=True, text=True)
    text_output = output.stdout.strip()
    return output.returncode, text_output


def check_conda_installed():
    if os.name == 'nt':
        check_conda_command = 'where conda'
    else:
        check_conda_command = 'which conda'
    retcode, text = run_command(check_conda_command)
    if retcode != 0:
        raise OSError('Conda installation not found.')


if __name__ == '__main__':
    raise OSError('Not meant to be run directly. Import functions instead.')
