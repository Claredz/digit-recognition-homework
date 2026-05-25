import numpy as np
from PIL import Image

from src.predict import preprocess_digit_image


def test_preprocess_digit_image_size_and_value_range():
    array = np.zeros((20, 20), dtype=np.uint8)
    array[5:15, 8:12] = 255

    processed = preprocess_digit_image(Image.fromarray(array).convert("L"), image_size=28)
    values = np.asarray(processed)

    assert processed.size == (28, 28)
    assert values.min() >= 0
    assert values.max() <= 255
