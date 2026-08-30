from pathlib import Path

import numpy as np
from PIL import Image


def image_load(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Returns the Image as a numpy array in (r,g,b) format and Image size as in height x width as a tuple (rgb, (height, width))."""
    with Image.open(path) as im:
        pixels = im.convert("RGB")
        return (np.array(pixels), (im.height, im.width))
