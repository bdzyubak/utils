from typing import Union, Optional

import pandas as pd
from sklearn.model_selection import train_test_split


def pick_columns_trim_name(df: pd.DataFrame, str_pattern: str) -> pd.DataFrame:
    # Select columns from data frame using a pattern, then drop that repeated pattern from column names in the new
    # dataframe.
    # Use case: pick Accuracy: method columns, and the use Accuracy in title and just the method name in legend
    columns = [name for name in df.columns if name.startswith(str_pattern)]
    map_dict = dict()
    for column in columns:
        column_reformatted = column.replace(str_pattern, '')
        left = 0
        right = len(column_reformatted) - 1
        while left <= right:
            if not column_reformatted[left].isalnum() or column_reformatted[left] == ' ':
                left += 1
            if not column_reformatted[right].isalnum() or column_reformatted[left] == ' ':
                right -= 1
            else:
                break
        if left == right:
            raise ValueError('Change pattern - all characters post pattern are not alphanumeric and were dropped.')
        map_dict[column] = column_reformatted[left:(right+1)]

    df_acc = df[columns].rename(columns=map_dict)
    return df_acc


def set_display_rows_cols(max_rows: int = 20, max_columns: int = 20, max_width: int = 1000):
    # Sets how many rows and columns to display when looking at the pandas dataframe e.g. with df.head()
    # Use this at the top of a data analysis script
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_columns', max_columns)
    pd.set_option('display.width', max_width)


def do_train_val_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train, df_test_val = train_test_split(df, train_size=0.6)
    df_val, df_test = train_test_split(df_test_val, train_size=0.5)
    return df_test, df_train, df_val


def read_dataframe(file_path: str, nrows=None) -> pd.DataFrame:
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, nrows=nrows)
    elif file_path.endswith('.tsv'):
        df = pd.read_csv(file_path, nrows=nrows, sep='\t')
    else:
        raise ValueError(f'Unsupported file type: {file_path}')
    return df


def is_close(a: Union[int, float], b: Union[int, float], abs_tol=1e-4) -> bool:
    return abs(a - b) <= abs_tol
# def is_close(a, b, rel_tol=1e-4, abs_tol=1e-4):
    # return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
