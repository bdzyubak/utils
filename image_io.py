from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image


def jpg_to_tensor(path: Union[Path, str]):
    image = torch.tensor(np.array(Image.open(path)), dtype=torch.float32)
    return image
