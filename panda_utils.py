import pandas as pd


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


def set_display_rows_cols(max_rows=20, max_columns=20, max_width=1000):
    # Sets how many rows and columns to display when looking at the pandas dataframe e.g. with df.head()
    # Use this at the top of a data analysis script
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_columns', max_columns)
    pd.set_option('display.width', max_width)
