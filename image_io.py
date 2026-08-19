import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append("..")


def image_load(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        pixels = im.convert("RGB")
        return np.array(pixels)


def save_pixels(pixels: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = pixels.reshape(-1, 3)
    np.save(path, pixels)


def load_pixels(path: Path) -> np.ndarray:
    return np.load(path)


if __name__ == "__main__":
    test = image_load("data/cat.JPG")
    save_pixels(test, "data/results/cat")

    print(test)
