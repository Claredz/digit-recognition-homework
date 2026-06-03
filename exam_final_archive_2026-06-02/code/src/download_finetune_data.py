from __future__ import annotations

import argparse
import csv
import json
import shutil
import ssl
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

HASYV2_URL = "https://zenodo.org/record/259444/files/HASYv2.tar.bz2?download=1"
CHARS74K_ENGLISH_HND_URL = "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz"
PENBASED_TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra"
PENBASED_TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}


def _download(url: str, output_path: Path, force: bool = False) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        return output_path
    print(f"Downloading {url} -> {output_path}", flush=True)
    try:
        urllib.request.urlretrieve(url, output_path)
    except Exception:
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=context) as response, output_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    return output_path


def _extract_tar(archive_path: Path, output_dir: Path, marker_name: str, force: bool = False) -> Path:
    marker = output_dir / marker_name
    if marker.exists() and not force:
        return output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as archive:
        archive.extractall(output_dir)
    marker.write_text("ok", encoding="utf-8")
    return output_dir


def _clear_digit_output(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for label in range(10):
        label_dir = output_dir / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)


def _count_digit_folder(output_dir: Path) -> dict[str, int]:
    return {
        str(label): sum(1 for path in (output_dir / str(label)).glob("*") if path.is_file())
        for label in range(10)
    }


def prepare_hasyv2(project_root: Path, force: bool = False) -> dict:
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw_external" / "hasyv2"
    archive_path = raw_dir / "HASYv2.tar.bz2"
    extract_dir = raw_dir / "extracted"
    output_dir = data_dir / "hasyv2_digits"

    _download(HASYV2_URL, archive_path, force=force)
    _extract_tar(archive_path, extract_dir, ".extracted", force=force)
    _clear_digit_output(output_dir)

    csv_candidates = list(extract_dir.rglob("hasy-data-labels.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"未找到 HASYv2 标签文件: {extract_dir}")
    labels_csv = csv_candidates[0]
    base_dir = labels_csv.parent
    copied = 0
    with labels_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            latex = (row.get("latex") or row.get("symbol") or "").strip()
            if latex not in {str(i) for i in range(10)}:
                continue
            path_value = row.get("path") or row.get("image") or row.get("filename")
            if not path_value:
                continue
            image_path = base_dir / path_value
            if not image_path.exists():
                image_path = extract_dir / path_value
            if not image_path.exists():
                continue
            target = output_dir / latex / f"hasyv2_{copied:06d}{image_path.suffix.lower()}"
            shutil.copy2(image_path, target)
            copied += 1

    counts = _count_digit_folder(output_dir)
    return {"name": "hasyv2_digits", "output_dir": str(output_dir), "num_images": copied, "counts": counts}


def prepare_chars74k(project_root: Path, force: bool = False) -> dict:
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw_external" / "chars74k"
    archive_path = raw_dir / "EnglishHnd.tgz"
    extract_dir = raw_dir / "extracted"
    output_dir = data_dir / "chars74k_digits"

    _download(CHARS74K_ENGLISH_HND_URL, archive_path, force=force)
    _extract_tar(archive_path, extract_dir, ".extracted", force=force)
    _clear_digit_output(output_dir)

    copied = 0
    for label in range(10):
        sample_name = f"Sample{label + 1:03d}"
        sample_dirs = [path for path in extract_dir.rglob(sample_name) if path.is_dir()]
        for sample_dir in sample_dirs:
            for image_path in sorted(sample_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                    target = output_dir / str(label) / f"chars74k_{label}_{copied:06d}{image_path.suffix.lower()}"
                    shutil.copy2(image_path, target)
                    copied += 1

    counts = _count_digit_folder(output_dir)
    return {"name": "chars74k_digits", "output_dir": str(output_dir), "num_images": copied, "counts": counts}


def _read_penbased_rows(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = [item.strip() for item in line.strip().split(",") if item.strip()]
            if len(parts) != 17:
                continue
            values = [int(float(item)) for item in parts]
            coords = values[:16]
            label = values[16]
            if 0 <= label <= 9:
                rows.append((coords, label))
    return rows


def _render_penbased(coords: list[int], image_size: int = 28, line_width: int = 2) -> Image.Image:
    points = []
    for index in range(0, len(coords), 2):
        x = int(round(2 + coords[index] / 100 * (image_size - 5)))
        y = int(round(2 + (100 - coords[index + 1]) / 100 * (image_size - 5)))
        points.append((x, y))
    image = Image.new("L", (image_size, image_size), color=0)
    draw = ImageDraw.Draw(image)
    if len(points) >= 2:
        draw.line(points, fill=255, width=line_width, joint="curve")
    for x, y in points:
        radius = max(1, line_width // 2)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    return image.filter(Image.Resampling.BILINEAR) if False else image


def prepare_penbased(project_root: Path, force: bool = False, widths: tuple[int, ...] = (1, 2, 3)) -> dict:
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw_external" / "penbased"
    train_path = raw_dir / "pendigits.tra"
    test_path = raw_dir / "pendigits.tes"
    output_dir = data_dir / "penbased_rendered"

    _download(PENBASED_TRAIN_URL, train_path, force=force)
    _download(PENBASED_TEST_URL, test_path, force=force)
    _clear_digit_output(output_dir)

    rows = _read_penbased_rows(train_path) + _read_penbased_rows(test_path)
    written = 0
    for row_index, (coords, label) in enumerate(rows):
        for width in widths:
            image = _render_penbased(coords, line_width=width)
            image.save(output_dir / str(label) / f"penbased_{row_index:06d}_w{width}.png")
            written += 1

    counts = _count_digit_folder(output_dir)
    return {"name": "penbased_rendered", "output_dir": str(output_dir), "num_images": written, "counts": counts}


def prepare_all(project_root: Path, force: bool = False, skip_hasyv2: bool = False, skip_chars74k: bool = False, skip_penbased: bool = False):
    project_root = Path(project_root)
    results = []
    errors = []
    tasks = [
        ("hasyv2", prepare_hasyv2, skip_hasyv2),
        ("chars74k", prepare_chars74k, skip_chars74k),
        ("penbased", prepare_penbased, skip_penbased),
    ]
    for name, function, skip in tasks:
        if skip:
            continue
        try:
            results.append(function(project_root, force=force))
        except Exception as exc:
            errors.append({"name": name, "error": str(exc)})
            print(f"WARNING: {name} preparation failed: {exc}", flush=True)

    manifest = {"results": results, "errors": errors}
    manifest_path = project_root / "data" / "finetune_datasets_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Download and prepare western digit fine-tuning datasets")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-hasyv2", action="store_true")
    parser.add_argument("--skip-chars74k", action="store_true")
    parser.add_argument("--skip-penbased", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = prepare_all(
        args.project_root.resolve(),
        force=args.force,
        skip_hasyv2=args.skip_hasyv2,
        skip_chars74k=args.skip_chars74k,
        skip_penbased=args.skip_penbased,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
