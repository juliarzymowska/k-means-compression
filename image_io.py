import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append("..")


def image_load(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Returns the Image as a numpy array in (r,g,b) format and Image size as in height x width as a tuple (rgb, (height, width))."""
    with Image.open(path) as im:
        pixels = im.convert("RGB")
        return (np.array(pixels), (im.height, im.width))


def save_pixels(pixels: np.ndarray, size: tuple[int, int], path: Path) -> None:
    """Save the Image (r,g,b) pixels and height x width as binary numpy file .npy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, pixels=pixels.reshape(-1, 3), size=np.array(size))


def load_pixels(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Load pixels + original (height, width) saved by save_pixels."""
    data = np.load(path)
    return data["pixels"], tuple(data["size"])


if __name__ == "__main__":
    pixels, size = image_load("data/cat.JPG")
    save_pixels(pixels, size, "data/results/cat")

    print(pixels)
