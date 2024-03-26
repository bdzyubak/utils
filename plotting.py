import pandas as pd
from pathlib import Path
from typing import Optional, Union

import seaborn as sns
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (14, 8)  # must set at the top
plt.rcParams.update({'font.size': 22})  # must set at the top
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from six import StringIO
from IPython.display import Image
import pydotplus

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


def plot_decision_tree_architecture(clf: DecisionTreeClassifier, feature_names: list[str], file_path: Union[Path, str]):
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_names, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(str(file_path))
    Image(graph.create_png())
