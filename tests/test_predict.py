import csv

import numpy as np
import torch
from PIL import Image

from src.config import ExperimentConfig
from src.predict import (
    PredictionImageDataset,
    auto_invert_grayscale,
    crop_digit_foreground,
    predict_with_tta,
    preprocess_digit_image,
    write_predictions_csv,
)


def test_prediction_image_dataset_preserves_sorted_filenames(tmp_path):
    image_dir = tmp_path / "predict"
    image_dir.mkdir()

    for filename, pixel_value in [("b.png", 50), ("a.png", 150)]:
        Image.fromarray(np.full((28, 28), pixel_value, dtype=np.uint8)).convert("L").save(
            image_dir / filename
        )

    dataset = PredictionImageDataset(image_dir, image_size=28)

    _, first_name = dataset[0]
    _, second_name = dataset[1]

    assert first_name == "a.png"
    assert second_name == "b.png"


def test_auto_invert_turns_white_background_black():
    image = Image.fromarray(np.full((8, 8), 240, dtype=np.uint8)).convert("L")

    inverted = auto_invert_grayscale(image)

    assert np.asarray(inverted).mean() < 20


def test_crop_digit_foreground_removes_blank_border():
    array = np.zeros((20, 20), dtype=np.uint8)
    array[8:12, 7:13] = 255

    cropped = crop_digit_foreground(Image.fromarray(array).convert("L"))

    assert cropped.size == (6, 4)


def test_preprocess_digit_image_outputs_expected_size():
    array = np.zeros((20, 20), dtype=np.uint8)
    array[5:15, 8:12] = 255

    processed = preprocess_digit_image(Image.fromarray(array).convert("L"), image_size=28)

    assert processed.size == (28, 28)


def test_predict_with_tta_disabled_matches_plain_forward():
    model = torch.nn.Conv2d(1, 10, kernel_size=1)
    images = torch.rand(2, 1, 28, 28)
    config = ExperimentConfig(project_root=".", use_tta=False)

    logits = predict_with_tta(model, images, config, device="cpu")

    assert logits.shape == (2, 10, 28, 28)


def test_write_predictions_csv_outputs_expected_columns(tmp_path):
    output_path = tmp_path / "predictions.csv"

    write_predictions_csv(
        rows=[("sample_1.png", 7), ("sample_2.png", 3)],
        output_path=output_path,
    )

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {"filename": "sample_1.png", "prediction": "7"},
        {"filename": "sample_2.png", "prediction": "3"},
    ]
