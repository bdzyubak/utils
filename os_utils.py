from pathlib import Path
import shutil
import subprocess
from typing import Union, Tuple
import re


def run_command(command: Union[str, list], verbose: bool = False) -> Tuple[int, str]:
    output = subprocess.run(command, capture_output=True, text=True)
    text_output = output.stdout
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
    if isinstance(path, str):
        path = Path(path)

    remove_dir(path, accept_fail_to_remove)

    path.mkdir(exist_ok=True, parents=True)


def remove_dir(path: Union[str, Path], accept_fail_to_remove: bool = False):
    if path.exists():
        try:
            shutil.rmtree(path)
        except PermissionError as e:
            if not accept_fail_to_remove:
                raise e
            else:
                print(f'WARNING: Failed to remove directory {path}')


def filename_to_title(file_path: Union[Path, str]):
    if isinstance(file_path, str):
        file_path = Path(file_path)
    filename = file_path.name.replace('_', ' ').strip()
    words = [name[0].upper()+name[1:] for name in filename.split(' ')]
    return ' '.join(words)


def endswith_list(string: str, suffix_list: list) -> bool:
    # Check if string ends with any suffix in list
    ends_with_any = list(filter(string.endswith, suffix_list)) != []
    return ends_with_any
