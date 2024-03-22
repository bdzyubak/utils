import pandas as pd
from pathlib import Path
from typing import Optional, Union

import seaborn as sns
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (14, 8)  # must set at the top
plt.rcParams.update({'font.size': 22})  # must set at the top

from utils.os_utils import filename_to_title


def lineplot(df: pd.DataFrame, save_file: Optional[Path] = None, xlabel: Optional[str] = None,
             ylabel: Optional[str] = None, legend_outside_plot: bool = True):
    plt.figure()
    color_palette = sns.color_palette()
    # Skip first column which is the index
    plot = sns.lineplot(data=df[df.columns[1:]], palette=color_palette, legend=True)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    title = filename_to_title(save_file)
    plt.title(title)
    if legend_outside_plot:
        sns.move_legend(plot.axes, 'center right')
    else:
        sns.move_legend(plot.axes, 'center right', bbox_to_anchor=(1, 0.5), ncol=1)

    plt.savefig(save_file)
    plt.show()
