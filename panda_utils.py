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
