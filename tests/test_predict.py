import csv

import numpy as np
from PIL import Image

from src.predict import PredictionImageDataset, write_predictions_csv


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
