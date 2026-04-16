import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def folder_digit_dataset(tmp_path):
    root = tmp_path / "digits"
    for label, pixel_value in [(0, 30), (1, 220)]:
        label_dir = root / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in range(4):
            arr = np.full((20, 20), pixel_value + index, dtype=np.uint8)
            image = Image.fromarray(arr).convert("L")
            image.save(label_dir / f"{label}_{index}.png")
    return root
