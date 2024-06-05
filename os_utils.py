from pathlib import Path
import shutil
import subprocess
from typing import Union, Tuple
import re

import mlflow
import psutil


def run_command(command: str, verbose: bool = False) -> Tuple[int, str]:
    """
    A wrapper to run a command in the command line and interpret the output. OS agnostic
    Args:
        command: a command to run in Linux terminal or Windows command prompt
        verbose: Print command return info
    Returns:
        None
    """

    output = subprocess.run(command, capture_output=True)
    text_output = output.stdout.decode()
    # Remove conda escape characters, if present
    ansi_cleaned = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

    text_output = ansi_cleaned.sub('', text_output)
    print(text_output)
    if output.returncode != 0:
        raise OSError(f'Could not execute command {command}')

    if verbose:
        print(f'Command: {command}')
        print(f'Result: {output.returncode}: {text_output}')
    return output.returncode, text_output


def get_file(search_path: Union[str, Path], mask: str = '*') -> Path:
    """
    A function to get a single file at target path. Checks for too many or non-existant files
    Args:
        search_path: Directory to look in. If file - return self
        mask: Mask to search with. Can be exact file name, extension, or a more complicated wildcard expression

    Returns:
        file_path: Path to single located file
    """
    if not isinstance(search_path, Path):
        search_path = Path(search_path)

    if search_path.is_file():
        return search_path

    files = list(search_path.glob(mask))
    if not files:
        raise OSError(f'File with mask {mask} not found at {search_path}.')
    if len(files) > 1:
        raise OSError(f'Multiple files match mask {mask} at {search_path}')
    file_path: Path = files[0]
    return file_path


def make_fresh_dir(path: Union[str, Path], accept_fail_to_remove: bool = False):
    """
    A wrapper to remove an existing directory, and make a clean one with the same name
    Args:
        path: Path including the directory name
        accept_fail_to_remove: Ignore write locks causing inability to remove completely. se only if mixing artifact
        carries minimal risk
    """
    if isinstance(path, str):
        path = Path(path)

    remove_dir(path, accept_fail_to_remove)

    path.mkdir(exist_ok=True, parents=True)


def remove_dir(path: Union[str, Path], accept_fail_to_remove: bool = False):
    """
    Wrapper for shutil rmtree to accept if the tree is missing.
    Args:
        path: Path to directory to remove
        accept_fail_to_remove: Accept inability to remove completely due to file lock. Use only if mixing artifact
        carries minimal risk
    """
    if path.exists():
        try:
            shutil.rmtree(path)
        except PermissionError as e:
            if not accept_fail_to_remove:
                raise e
            else:
                print(f'WARNING: Failed to remove directory {path}')


def filename_to_title(file_path: Union[Path, str]):
    """
    Converter of filename with underscores to pretty chart name with spaces and first letter capitalization, to minimize
    mismatch errors and confusion
    Args:
        file_path: Name of file to which the artifact will be saved.
    Returns:
        None
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    filename = file_path.name.replace('_', ' ').strip()
    words = [name[0].upper()+name[1:] for name in filename.split(' ')]
    return ' '.join(words)


def endswith_list(string: str, suffix_list: list) -> bool:
    # Check if string ends with any suffix in list
    ends_with_any = list(filter(string.endswith, suffix_list)) != []
    return ends_with_any


def startswith_list(string: str, suffix_list: list) -> bool:
    # Check if string ends with any suffix in list
    starts_with_any = list(filter(string.startswith, suffix_list)) != []
    return starts_with_any


def get_memory_use(code_point: str = '', log_to_mlflow: bool = False):
    """
    Memory profiling function. Use at multiple points or iterations to check if there is a leak
    Args:
        code_point: String message to use to indicate where this function is being run
        log_to_mlflow: Flag to log as a metric on the mlflow server
    """
    memory_usage = psutil.virtual_memory().used
    memory_usage = round(memory_usage / 1e9, 1)
    print(f"Memory use {code_point} {memory_usage} GB")
    if log_to_mlflow and not mlflow.active_run():
        raise OSError('No mlflow run active.')

    if log_to_mlflow:
        mlflow.log_metric(f'memory_usage_{code_point}', memory_usage)
    return memory_usage


def str_to_path(path: Union[str, Path]):
    if isinstance(path, str):
        path = Path(path)
    return path


def number_to_order_of_magnitude_string(param_number: int) -> str:
    # Specify order of magnitude, keep one decimal point beyond it
    if param_number > 1000000:
        number_as_string = f"{round(param_number / 100000) / 10} M"
    elif param_number > 1000:
        number_as_string = f"{round(param_number / 100) / 10} K"
    else:
        number_as_string = str(param_number)
    return number_as_string


def get_matched_files(path_source1, path_source2, extensions=None):
    if extensions is None:
        extensions = ['*.txt', '*.txt']
    elif not isinstance(extensions, list):
        extensions = [extensions, extensions]

    files_source1 = list(path_source1.glob(extensions[0]))
    files_source2 = list(path_source2.glob(extensions[1]))

    source2_stems = [path.stem for path in files_source2]
    files_source1 = sorted([path for path in files_source1 if path.stem in source2_stems])

    source1_stems = [path.stem for path in files_source1]
    files_source2 = sorted([path for path in files_source2 if path.stem in source1_stems])
    return files_source1, files_source2
