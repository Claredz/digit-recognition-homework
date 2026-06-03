from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def auto_invert_grayscale(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    array = np.asarray(gray, dtype=np.uint8)
    corner_pixels = np.concatenate(
        [array[:3, :].ravel(), array[-3:, :].ravel(), array[:, :3].ravel(), array[:, -3:].ravel()]
    )
    background_level = float(np.median(corner_pixels)) if corner_pixels.size else float(array.mean())
    if background_level > 127.0:
        array = 255 - array
    return Image.fromarray(array, mode="L")


def cleanup_background(image: Image.Image, threshold: int = 12) -> Image.Image:
    array = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    array[array < threshold] = 0
    return Image.fromarray(array, mode="L")


def crop_digit_foreground(image: Image.Image, threshold: int = 20) -> Image.Image:
    array = np.asarray(image.convert("L"), dtype=np.uint8)
    if array.size == 0:
        return Image.new("L", (1, 1), color=0)
    dynamic_threshold = max(threshold, int(array.max() * 0.10))
    ys, xs = np.where(array > dynamic_threshold)
    if len(xs) == 0 or len(ys) == 0:
        return image.convert("L")
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    return image.crop((left, top, right, bottom))


def resize_digit_body(image: Image.Image, image_size: int = 28, body_size: int = 20) -> Image.Image:
    image = image.convert("L")
    width, height = image.size
    if width <= 0 or height <= 0:
        return Image.new("L", (image_size, image_size), color=0)
    scale = min(body_size / width, body_size / height)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized = image.resize(new_size, Image.Resampling.BILINEAR)
    canvas = Image.new("L", (image_size, image_size), color=0)
    offset = ((image_size - new_size[0]) // 2, (image_size - new_size[1]) // 2)
    canvas.paste(resized, offset)
    return canvas


def center_by_mass(image: Image.Image, image_size: int = 28) -> Image.Image:
    array = np.asarray(image.convert("L"), dtype=np.float32)
    total = float(array.sum())
    if total <= 0:
        return Image.fromarray(array.astype(np.uint8), mode="L")

    ys, xs = np.indices(array.shape)
    center_x = float((xs * array).sum() / total)
    center_y = float((ys * array).sum() / total)
    target = (image_size - 1) / 2.0
    shift_x = int(round(target - center_x))
    shift_y = int(round(target - center_y))

    shifted = np.zeros((image_size, image_size), dtype=np.uint8)
    source = np.asarray(image.convert("L"), dtype=np.uint8)
    src_y0 = max(0, -shift_y)
    src_y1 = min(image_size, image_size - shift_y)
    src_x0 = max(0, -shift_x)
    src_x1 = min(image_size, image_size - shift_x)
    dst_y0 = max(0, shift_y)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = max(0, shift_x)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    if src_y1 > src_y0 and src_x1 > src_x0:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = source[src_y0:src_y1, src_x0:src_x1]
    return Image.fromarray(shifted, mode="L")


def preprocess_to_mnist_style_image(
    image: Image.Image,
    image_size: int = 28,
    body_size: int = 20,
    auto_invert: bool = True,
    foreground_threshold: int = 20,
) -> Image.Image:
    gray = image.convert("L")
    if auto_invert:
        gray = auto_invert_grayscale(gray)
    cleaned = cleanup_background(gray)
    cropped = crop_digit_foreground(cleaned, threshold=foreground_threshold)
    resized = resize_digit_body(cropped, image_size=image_size, body_size=body_size)
    return center_by_mass(resized, image_size=image_size)


def preprocess_to_mnist_style(
    image: Image.Image,
    image_size: int = 28,
    body_size: int = 20,
    auto_invert: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    processed = preprocess_to_mnist_style_image(
        image,
        image_size=image_size,
        body_size=body_size,
        auto_invert=auto_invert,
    )
    array = np.asarray(processed, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0)
    if normalize:
        tensor = (tensor - 0.5) / 0.5
    return tensor
