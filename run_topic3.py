from __future__ import annotations

import csv
import math
import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import color, exposure, feature, filters, io, measure, util


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "Εργαστηριακή Άσκηση - Μέρος Β - 2025-2026" / "Υλικό Θέμα 3"
LOW_DIR = DATA_DIR / "low"
HIGH_DIR = DATA_DIR / "high"
OUTPUT_DIR = ROOT_DIR / "topic3_low_light_solution" / "results"

GAMMA_VALUES = [0.4, 0.6, 0.8, 1.2]
CLAHE_VARIANTS = [
    ("CLAHE clip=0.01 tile=8", 0.01, 8),
    ("CLAHE clip=0.05 tile=8", 0.05, 8),
    ("CLAHE clip=0.10 tile=8", 0.10, 8),
    ("CLAHE clip=0.02 tile=16", 0.02, 16),
    ("CLAHE clip=0.02 tile=32", 0.02, 32),
]
RETINEX_SIGMAS = [10.0, 20.0, 30.0]
UNSHARP_AMOUNTS = [0.5, 1.0, 1.5]


@dataclass
class MethodResult:
    name: str
    image: np.ndarray


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def clip01(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0)


def make_rng(*parts: str) -> np.random.Generator:
    seed_text = "|".join(parts)
    seed = zlib.crc32(seed_text.encode("utf-8")) & 0xFFFFFFFF
    return np.random.default_rng(seed)


def load_rgb_image(path: Path) -> np.ndarray:
    image = io.imread(path)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = util.img_as_float32(image)
    return clip01(image)


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    io.imsave(path, util.img_as_ubyte(clip01(image)), check_contrast=False)


def gray(image: np.ndarray) -> np.ndarray:
    return color.rgb2gray(image)


def apply_on_value_channel(image: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    hsv = color.rgb2hsv(image)
    hsv[..., 2] = clip01(fn(hsv[..., 2]))
    return clip01(color.hsv2rgb(hsv))


def linear_contrast_stretch_channel(channel: np.ndarray) -> np.ndarray:
    cmin = float(channel.min())
    cmax = float(channel.max())
    if cmax - cmin < 1e-8:
        return channel.copy()
    return (channel - cmin) / (cmax - cmin)


def linear_contrast_stretch(image: np.ndarray) -> np.ndarray:
    return apply_on_value_channel(image, linear_contrast_stretch_channel)


def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    return apply_on_value_channel(image, lambda v: np.power(np.clip(v, 0.0, 1.0), gamma))


def log_transform(image: np.ndarray) -> np.ndarray:
    return apply_on_value_channel(
        image,
        lambda v: np.log1p(255.0 * np.clip(v, 0.0, 1.0)) / np.log(256.0),
    )


def global_hist_equalization(image: np.ndarray) -> np.ndarray:
    return apply_on_value_channel(image, exposure.equalize_hist)


def clahe(image: np.ndarray, clip_limit: float = 0.02, tile_size: int = 8) -> np.ndarray:
    return apply_on_value_channel(
        image,
        lambda v: exposure.equalize_adapthist(
            v,
            clip_limit=clip_limit,
            kernel_size=(tile_size, tile_size),
        ),
    )


def mean_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    return clip01(ndi.uniform_filter(image, size=(size, size, 1)))


def gaussian_filter_rgb(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return clip01(ndi.gaussian_filter(image, sigma=(sigma, sigma, 0)))


def median_filter_rgb(image: np.ndarray, size: int = 3) -> np.ndarray:
    filtered = np.empty_like(image)
    for channel in range(image.shape[2]):
        filtered[..., channel] = ndi.median_filter(image[..., channel], size=size)
    return clip01(filtered)


def add_gaussian_noise(
    image: np.ndarray,
    variance: float = 0.002,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    return clip01(util.random_noise(image, mode="gaussian", var=variance, rng=rng))


def add_salt_pepper_noise(
    image: np.ndarray,
    amount: float = 0.02,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    return clip01(util.random_noise(image, mode="s&p", amount=amount, rng=rng))


def laplacian_sharpen(image: np.ndarray, amount: float = 0.35) -> np.ndarray:
    sharpened = np.empty_like(image)
    for channel in range(image.shape[2]):
        lap = filters.laplace(image[..., channel], ksize=3)
        sharpened[..., channel] = image[..., channel] - amount * lap
    return clip01(sharpened)


def unsharp_masking(image: np.ndarray, amount: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    blurred = gaussian_filter_rgb(image, sigma=sigma)
    return clip01(image + amount * (image - blurred))


def simple_retinex(image: np.ndarray, sigma: float = 20.0, eps: float = 1e-3) -> np.ndarray:
    def _retinex(v: np.ndarray) -> np.ndarray:
        illumination = ndi.gaussian_filter(v, sigma=sigma)
        reflectance = np.log(v + eps) - np.log(illumination + eps)
        return exposure.rescale_intensity(reflectance, in_range="image", out_range=(0.0, 1.0))

    return apply_on_value_channel(image, _retinex)


def pipeline_simple(image: np.ndarray) -> np.ndarray:
    return linear_contrast_stretch(gamma_correction(image, gamma=0.6))


def pipeline_clahe_unsharp(image: np.ndarray) -> np.ndarray:
    denoised = median_filter_rgb(image, size=3)
    enhanced = clahe(denoised, clip_limit=0.02, tile_size=8)
    return unsharp_masking(enhanced, amount=1.0, sigma=1.0)


def pipeline_retinex(image: np.ndarray, sigma: float = 20.0) -> np.ndarray:
    retinex = simple_retinex(gaussian_filter_rgb(image, sigma=0.8), sigma=sigma)
    retinex = gaussian_filter_rgb(retinex, sigma=0.8)
    mixed = 0.55 * linear_contrast_stretch(retinex) + 0.45 * gamma_correction(image, gamma=0.8)
    return clip01(mixed)


def edge_map(image: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    return feature.canny(gray(image), sigma=sigma)


def histogram_and_cdf(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = util.img_as_ubyte(gray(image))
    hist, _ = np.histogram(values, bins=256, range=(0, 255))
    pdf = hist.astype(np.float64) / max(hist.sum(), 1)
    cdf = np.cumsum(pdf)
    return hist, cdf


def compute_basic_stats(image: np.ndarray) -> dict[str, float]:
    g = gray(image)
    mean = float(g.mean() * 255.0)
    variance = float(g.var() * (255.0**2))
    contrast = float(g.std() * 255.0)
    entropy = float(measure.shannon_entropy(util.img_as_ubyte(g)))
    max_i = float(g.max())
    min_i = float(g.min())
    michelson = 0.0 if max_i + min_i < 1e-8 else float((max_i - min_i) / (max_i + min_i))
    edges = edge_map(image)
    edge_pixels = int(edges.sum())
    edge_ratio = float(edge_pixels / edges.size)
    return {
        "mean_brightness": mean,
        "variance": variance,
        "contrast_std": contrast,
        "entropy": entropy,
        "michelson_contrast": michelson,
        "edge_pixels": edge_pixels,
        "edge_ratio": edge_ratio,
    }


def compute_reference_metrics(reference: np.ndarray | None, image: np.ndarray) -> dict[str, float]:
    if reference is None:
        return {"mse": math.nan, "psnr_db": math.nan}
    ref255 = reference.astype(np.float64) * 255.0
    img255 = image.astype(np.float64) * 255.0
    mse = float(np.mean((ref255 - img255) ** 2))
    psnr = float("inf") if mse == 0 else float(10.0 * math.log10((255.0**2) / mse))
    return {"mse": mse, "psnr_db": psnr}


def compute_edge_quality(reference: np.ndarray | None, image: np.ndarray) -> dict[str, float]:
    if reference is None:
        return {
            "edge_precision": math.nan,
            "edge_recall": math.nan,
            "edge_f1": math.nan,
            "edge_iou": math.nan,
        }

    ref_edges = edge_map(reference)
    img_edges = edge_map(image)
    tp = float(np.logical_and(ref_edges, img_edges).sum())
    fp = float(np.logical_and(~ref_edges, img_edges).sum())
    fn = float(np.logical_and(ref_edges, ~img_edges).sum())
    union = float(np.logical_or(ref_edges, img_edges).sum())

    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    iou = 0.0 if union == 0 else tp / union
    return {
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "edge_iou": iou,
    }


def add_target_metrics(row: dict[str, float | str], target_name: str, target: np.ndarray | None, image: np.ndarray) -> None:
    pixel_metrics = compute_reference_metrics(target, image)
    edge_metrics = compute_edge_quality(target, image)
    row[f"mse_to_{target_name}"] = pixel_metrics["mse"]
    row[f"psnr_to_{target_name}"] = pixel_metrics["psnr_db"]
    row[f"edge_precision_to_{target_name}"] = edge_metrics["edge_precision"]
    row[f"edge_recall_to_{target_name}"] = edge_metrics["edge_recall"]
    row[f"edge_f1_to_{target_name}"] = edge_metrics["edge_f1"]
    row[f"edge_iou_to_{target_name}"] = edge_metrics["edge_iou"]


def blank_axis(axis: plt.Axes, title: str, message: str) -> None:
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
    axis.axis("off")


def make_analysis_figure(name: str, low_image: np.ndarray, ref_image: np.ndarray | None, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    low_stats = compute_basic_stats(low_image)
    g = gray(low_image)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[0, 0].imshow(low_image)
    axes[0, 0].set_title(f"{name} low")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(g, cmap="gray")
    axes[0, 1].set_title("Grayscale")
    axes[0, 1].axis("off")

    axes[0, 2].hist((g * 255).ravel(), bins=256, color="dimgray")
    axes[0, 2].set_title("Luminance histogram")
    axes[0, 2].set_xlim(0, 255)

    for idx, color_name in enumerate(["red", "green", "blue"]):
        axes[1, 0].hist((low_image[..., idx] * 255).ravel(), bins=256, alpha=0.45, color=color_name)
    axes[1, 0].set_title("RGB histograms")
    axes[1, 0].set_xlim(0, 255)

    stats_text = (
        f"Mean: {low_stats['mean_brightness']:.2f}\n"
        f"Variance: {low_stats['variance']:.2f}\n"
        f"Contrast: {low_stats['contrast_std']:.2f}\n"
        f"Entropy: {low_stats['entropy']:.2f}\n"
        f"Michelson: {low_stats['michelson_contrast']:.3f}"
    )
    axes[1, 1].set_title("Low-light stats")
    axes[1, 1].text(0.05, 0.95, stats_text, ha="left", va="top", family="monospace", fontsize=10)
    axes[1, 1].axis("off")

    if ref_image is not None:
        axes[1, 2].imshow(ref_image)
        axes[1, 2].set_title("Reference high")
        axes[1, 2].axis("off")
    else:
        blank_axis(axes[1, 2], "Reference high", "No reference image")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_intensity_transform_figure(
    name: str,
    low_image: np.ndarray,
    ref_image: np.ndarray | None,
    results: list[MethodResult],
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    items: list[tuple[str, np.ndarray | None]] = [("Low", low_image), ("Reference", ref_image)]
    items.extend([(result.name, result.image) for result in results])
    for axis, (title, image) in zip(axes.ravel(), items):
        if image is None:
            blank_axis(axis, title, "No reference image")
        else:
            axis.imshow(image)
            axis.set_title(title)
            axis.axis("off")
    fig.suptitle(f"Intensity transforms for {name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_hist_eq_figure(name: str, low_image: np.ndarray, hist_eq_image: np.ndarray, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    low_hist, low_cdf = histogram_and_cdf(low_image)
    eq_hist, eq_cdf = histogram_and_cdf(hist_eq_image)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].imshow(low_image)
    axes[0, 0].set_title("Low image")
    axes[0, 0].axis("off")

    axes[0, 1].bar(np.arange(256), low_hist, width=1.0, color="dimgray")
    axes[0, 1].set_title("Histogram before")
    axes[0, 1].set_xlim(0, 255)

    axes[0, 2].plot(low_cdf, color="black")
    axes[0, 2].set_title("CDF before")
    axes[0, 2].set_xlim(0, 255)
    axes[0, 2].set_ylim(0, 1)

    axes[1, 0].imshow(hist_eq_image)
    axes[1, 0].set_title("Global HistEq")
    axes[1, 0].axis("off")

    axes[1, 1].bar(np.arange(256), eq_hist, width=1.0, color="steelblue")
    axes[1, 1].set_title("Histogram after")
    axes[1, 1].set_xlim(0, 255)

    axes[1, 2].plot(eq_cdf, color="navy")
    axes[1, 2].set_title("CDF after")
    axes[1, 2].set_xlim(0, 255)
    axes[1, 2].set_ylim(0, 1)

    fig.suptitle(f"Global histogram equalization for {name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_clahe_sweep_figure(name: str, low_image: np.ndarray, clahe_results: list[MethodResult], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    items = [MethodResult("Low", low_image), *clahe_results[:5]]
    for axis, result in zip(axes.ravel(), items):
        axis.imshow(result.image)
        axis.set_title(result.name)
        axis.axis("off")
    fig.suptitle(f"CLAHE parameter sweep for {name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_retinex_sweep_figure(name: str, low_image: np.ndarray, retinex_results: list[MethodResult], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    items = [MethodResult("Low", low_image), *retinex_results[:3]]
    for axis, result in zip(axes.ravel(), items):
        axis.imshow(result.image)
        axis.set_title(result.name)
        axis.axis("off")
    fig.suptitle(f"Retinex sigma sweep for {name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_sharpen_figure(name: str, base_image: np.ndarray, sharpen_results: list[MethodResult], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    items = [MethodResult("Pipeline 2 base", base_image), *sharpen_results[:3]]
    for axis, result in zip(axes.ravel(), items):
        axis.imshow(result.image)
        axis.set_title(result.name)
        axis.axis("off")
    fig.suptitle(f"Sharpening study for {name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_edge_figure(name: str, edge_results: list[MethodResult], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, result in zip(axes.ravel(), edge_results):
        axis.imshow(edge_map(result.image), cmap="gray")
        axis.set_title(f"Edges: {result.name}")
        axis.axis("off")
    fig.suptitle(f"Edge maps for {name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_noise_filter_figure(
    name: str,
    noise_label: str,
    low_image: np.ndarray,
    noisy_image: np.ndarray,
    filtered_results: list[MethodResult],
    ref_image: np.ndarray | None,
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    items: list[tuple[str, np.ndarray | None]] = [
        ("Original low", low_image),
        (f"{noise_label} noisy", noisy_image),
        (filtered_results[0].name, filtered_results[0].image),
        (filtered_results[1].name, filtered_results[1].image),
        (filtered_results[2].name, filtered_results[2].image),
        ("Reference high", ref_image),
    ]
    for axis, (title, image) in zip(axes.ravel(), items):
        if image is None:
            blank_axis(axis, title, "No reference image")
        else:
            axis.imshow(image)
            axis.set_title(title)
            axis.axis("off")
    fig.suptitle(f"{noise_label} denoising for {name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_noise_order_figure(
    name: str,
    noise_label: str,
    low_image: np.ndarray,
    noisy_image: np.ndarray,
    filtered_image: np.ndarray,
    denoise_then_enhance: np.ndarray,
    enhance_then_denoise: np.ndarray,
    ref_image: np.ndarray | None,
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    items: list[tuple[str, np.ndarray | None]] = [
        ("Original low", low_image),
        (f"{noise_label} noisy", noisy_image),
        ("Filtered", filtered_image),
        ("Denoise -> Enhance", denoise_then_enhance),
        ("Enhance -> Denoise", enhance_then_denoise),
        ("Reference high", ref_image),
    ]
    for axis, (title, image) in zip(axes.ravel(), items):
        if image is None:
            blank_axis(axis, title, "No reference image")
        else:
            axis.imshow(image)
            axis.set_title(title)
            axis.axis("off")
    fig.suptitle(f"{noise_label} processing order for {name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def sanitize_name(name: str) -> str:
    safe = name.replace(" ", "_").replace("=", "-").replace(".", "p")
    safe = safe.replace("(", "").replace(")", "").replace(":", "").replace(">", "to")
    return safe


def build_primary_methods(low_image: np.ndarray) -> list[MethodResult]:
    methods = [MethodResult("Linear stretch", linear_contrast_stretch(low_image))]
    methods.extend(
        MethodResult(f"Gamma {gamma}", gamma_correction(low_image, gamma=gamma))
        for gamma in GAMMA_VALUES
    )
    methods.append(MethodResult("Log transform", log_transform(low_image)))
    methods.append(MethodResult("Global HistEq", global_hist_equalization(low_image)))
    methods.extend(
        MethodResult(name, clahe(low_image, clip_limit=clip_limit, tile_size=tile_size))
        for name, clip_limit, tile_size in CLAHE_VARIANTS
    )
    methods.append(MethodResult("Pipeline 1", pipeline_simple(low_image)))
    methods.append(MethodResult("Pipeline 2", pipeline_clahe_unsharp(low_image)))
    methods.extend(
        MethodResult(f"Pipeline 3 sigma={int(sigma)}", pipeline_retinex(low_image, sigma=sigma))
        for sigma in RETINEX_SIGMAS
    )
    base = pipeline_clahe_unsharp(low_image)
    methods.append(MethodResult("Laplacian sharp", laplacian_sharpen(base, amount=0.25)))
    methods.extend(
        MethodResult(f"Unsharp k={amount}", unsharp_masking(base, amount=amount))
        for amount in UNSHARP_AMOUNTS
    )
    return methods


def method_lookup(methods: list[MethodResult]) -> dict[str, MethodResult]:
    return {method.name: method for method in methods}


def make_method_row(
    image_name: str,
    method_name: str,
    image: np.ndarray,
    low_reference: np.ndarray,
    high_reference: np.ndarray | None,
) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "image": image_name,
        "method": method_name,
        "reference_available": "yes" if high_reference is not None else "no",
    }
    row.update(compute_basic_stats(image))
    add_target_metrics(row, "low", low_reference, image)
    add_target_metrics(row, "high", high_reference, image)
    return row


def filter_variants(image: np.ndarray) -> list[MethodResult]:
    return [
        MethodResult("Mean filter", mean_filter(image, size=3)),
        MethodResult("Gaussian filter", gaussian_filter_rgb(image, sigma=1.0)),
        MethodResult("Median filter", median_filter_rgb(image, size=3)),
    ]


def best_order_filter_name(noise_type: str) -> str:
    return "Gaussian filter" if noise_type == "gaussian" else "Median filter"


def enhance_for_order(image: np.ndarray) -> np.ndarray:
    return clahe(image, clip_limit=0.02, tile_size=8)


def finite_mean(values: list[float]) -> float:
    finite_values = [value for value in values if not math.isnan(value) and not math.isinf(value)]
    if not finite_values:
        return math.nan
    return float(np.mean(finite_values))


def ordered_fieldnames(rows: list[dict[str, float | str]], preferred: list[str]) -> list[str]:
    all_keys = {key for row in rows for key in row}
    head = [key for key in preferred if key in all_keys]
    tail = sorted(all_keys - set(head))
    return head + tail


def write_csv(rows: list[dict[str, float | str]], path: Path, preferred: list[str]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_fieldnames(rows, preferred))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(rows: list[dict[str, float | str]], path: Path) -> None:
    ensure_dir(path.parent)
    grouped: dict[str, list[dict[str, float | str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)

    lines = [
        "# Topic 3 summary",
        "",
        "The table below aggregates all evaluated methods over the available low/high pairs.",
        "",
        "| Method | Avg PSNR to high (dB) | Avg MSE to high | Avg Brightness | Avg Contrast | Avg Entropy | Avg Edge F1 to high |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    excluded_from_ranking = {"Low input", "High reference"}
    best_method = None
    best_method_psnr = -float("inf")
    best_pipeline = None
    best_pipeline_psnr = -float("inf")

    for method, method_rows in sorted(grouped.items()):
        avg_psnr = finite_mean([float(row["psnr_to_high"]) for row in method_rows])
        avg_mse = finite_mean([float(row["mse_to_high"]) for row in method_rows])
        avg_brightness = finite_mean([float(row["mean_brightness"]) for row in method_rows])
        avg_contrast = finite_mean([float(row["contrast_std"]) for row in method_rows])
        avg_entropy = finite_mean([float(row["entropy"]) for row in method_rows])
        avg_edge_f1 = finite_mean([float(row["edge_f1_to_high"]) for row in method_rows])

        psnr_text = "N/A" if math.isnan(avg_psnr) else f"{avg_psnr:.2f}"
        mse_text = "N/A" if math.isnan(avg_mse) else f"{avg_mse:.2f}"
        edge_text = "N/A" if math.isnan(avg_edge_f1) else f"{avg_edge_f1:.3f}"
        lines.append(
            f"| {method} | {psnr_text} | {mse_text} | {avg_brightness:.2f} | {avg_contrast:.2f} | {avg_entropy:.2f} | {edge_text} |"
        )

        if method not in excluded_from_ranking and not math.isnan(avg_psnr) and avg_psnr > best_method_psnr:
            best_method_psnr = avg_psnr
            best_method = method
        if method.startswith("Pipeline") and not math.isnan(avg_psnr) and avg_psnr > best_pipeline_psnr:
            best_pipeline_psnr = avg_psnr
            best_pipeline = method

    lines.extend(
        [
            "",
            f"Best average PSNR among enhancement methods: `{best_method}` with `{best_method_psnr:.2f} dB`."
            if best_method is not None
            else "Best average PSNR among enhancement methods: `N/A`.",
            f"Best full pipeline: `{best_pipeline}` with `{best_pipeline_psnr:.2f} dB`."
            if best_pipeline is not None
            else "Best full pipeline: `N/A`.",
            "",
            "Suggested report angle:",
            "- Use the gamma sweep to discuss how gamma below 1 brightens dark regions and gamma above 1 darkens them.",
            "- Use the CLAHE sweep to discuss the tradeoff between local contrast and noise amplification.",
            "- Use the Retinex sweep to discuss how sigma changes local contrast and color stability.",
            "- Use `noise_metrics.csv` and `edge_metrics.csv` to support the denoising and edge-analysis sections.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def write_coverage_markdown(path: Path) -> None:
    ensure_dir(path.parent)
    lines = [
        "# Coverage Check",
        "",
        "| Topic 3 section | Output support |",
        "| --- | --- |",
        "| Part A - Low-light analysis | `analysis/`, `metrics.csv` |",
        "| Part B - Intensity transforms | `intensity/`, `metrics.csv` |",
        "| Part C - Histogram equalization | `hist_eq/`, `clahe_sweep/`, `metrics.csv` |",
        "| Part D - Denoising | `noise_filters/`, `noise_orders/`, `noise_metrics.csv` |",
        "| Part E - Sharpening | `sharpening/`, `metrics.csv` |",
        "| Part F - Full pipelines | `enhanced_images/`, `retinex_sweep/`, `metrics.csv`, `summary.md` |",
        "| Part G - Quantitative evaluation | `metrics.csv`, `summary.md` |",
        "| Part H - Edge impact | `edges/`, `edge_metrics.csv`, `metrics.csv` |",
        "",
        "If no `high` reference images are available, the script still runs and falls back to no-reference metrics.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    remove_tree(OUTPUT_DIR / "comparisons")
    remove_tree(OUTPUT_DIR / "noise_study")
    method_rows: list[dict[str, float | str]] = []
    noise_rows: list[dict[str, float | str]] = []
    edge_rows: list[dict[str, float | str]] = []

    low_paths = sorted(LOW_DIR.glob("*.png"))
    if not low_paths:
        raise SystemExit(f"No low-light images were found in {LOW_DIR}")

    analysis_dir = OUTPUT_DIR / "analysis"
    intensity_dir = OUTPUT_DIR / "intensity"
    hist_eq_dir = OUTPUT_DIR / "hist_eq"
    clahe_dir = OUTPUT_DIR / "clahe_sweep"
    retinex_dir = OUTPUT_DIR / "retinex_sweep"
    sharpening_dir = OUTPUT_DIR / "sharpening"
    edges_dir = OUTPUT_DIR / "edges"
    noise_filter_dir = OUTPUT_DIR / "noise_filters"
    noise_order_dir = OUTPUT_DIR / "noise_orders"
    outputs_dir = OUTPUT_DIR / "enhanced_images"

    for low_path in low_paths:
        name = low_path.stem
        high_path = HIGH_DIR / low_path.name
        ref_image = load_rgb_image(high_path) if high_path.exists() else None
        low_image = load_rgb_image(low_path)

        methods = build_primary_methods(low_image)
        lookup = method_lookup(methods)

        intensity_results = [
            lookup["Linear stretch"],
            lookup["Gamma 0.4"],
            lookup["Gamma 0.6"],
            lookup["Gamma 0.8"],
            lookup["Gamma 1.2"],
            lookup["Log transform"],
        ]
        clahe_results = [lookup[label] for label, _, _ in CLAHE_VARIANTS]
        retinex_results = [lookup[f"Pipeline 3 sigma={int(sigma)}"] for sigma in RETINEX_SIGMAS]
        sharpen_results = [
            lookup["Laplacian sharp"],
            lookup["Unsharp k=0.5"],
            lookup["Unsharp k=1.0"],
        ]
        edge_results = [
            MethodResult("Low", low_image),
            MethodResult("Pipeline 1", lookup["Pipeline 1"].image),
            MethodResult("Pipeline 2", lookup["Pipeline 2"].image),
            MethodResult("Pipeline 3 sigma=10", lookup["Pipeline 3 sigma=10"].image),
            MethodResult("Pipeline 3 sigma=20", lookup["Pipeline 3 sigma=20"].image),
            MethodResult("Global HistEq", lookup["Global HistEq"].image),
            MethodResult("CLAHE clip=0.05 tile=8", lookup["CLAHE clip=0.05 tile=8"].image),
            MethodResult("Unsharp k=1.0", lookup["Unsharp k=1.0"].image),
        ]

        make_analysis_figure(name, low_image, ref_image, analysis_dir / f"{name}_analysis.png")
        make_intensity_transform_figure(name, low_image, ref_image, intensity_results, intensity_dir / f"{name}_intensity.png")
        make_hist_eq_figure(name, low_image, lookup["Global HistEq"].image, hist_eq_dir / f"{name}_histeq.png")
        make_clahe_sweep_figure(name, low_image, clahe_results, clahe_dir / f"{name}_clahe.png")
        make_retinex_sweep_figure(name, low_image, retinex_results, retinex_dir / f"{name}_retinex.png")
        make_sharpen_figure(name, lookup["Pipeline 2"].image, sharpen_results, sharpening_dir / f"{name}_sharpening.png")
        make_edge_figure(name, edge_results, edges_dir / f"{name}_edges.png")

        for method in methods:
            save_rgb_image(outputs_dir / sanitize_name(method.name) / f"{name}.png", method.image)
            method_rows.append(make_method_row(name, method.name, method.image, low_image, ref_image))

        method_rows.append(make_method_row(name, "Low input", low_image, low_image, ref_image))
        if ref_image is not None:
            method_rows.append(make_method_row(name, "High reference", ref_image, low_image, ref_image))

        for edge_result in edge_results:
            edge_row: dict[str, float | str] = {
                "image": name,
                "method": edge_result.name,
                "reference_available": "yes" if ref_image is not None else "no",
            }
            edge_row.update(compute_basic_stats(edge_result.image))
            add_target_metrics(edge_row, "high", ref_image, edge_result.image)
            edge_rows.append(edge_row)

        noise_sources = {
            "gaussian": add_gaussian_noise(
                low_image,
                variance=0.004,
                rng=make_rng(name, "gaussian"),
            ),
            "salt_pepper": add_salt_pepper_noise(
                low_image,
                amount=0.03,
                rng=make_rng(name, "salt_pepper"),
            ),
        }
        for noise_type, noisy_image in noise_sources.items():
            filtered_results = filter_variants(noisy_image)
            make_noise_filter_figure(
                name,
                noise_type.replace("_", " ").title(),
                low_image,
                noisy_image,
                filtered_results,
                ref_image,
                noise_filter_dir / f"{name}_{noise_type}.png",
            )

            noisy_row: dict[str, float | str] = {
                "image": name,
                "noise_type": noise_type,
                "stage": "noisy",
                "filter_name": "none",
                "reference_available": "yes" if ref_image is not None else "no",
            }
            noisy_row.update(compute_basic_stats(noisy_image))
            add_target_metrics(noisy_row, "low", low_image, noisy_image)
            add_target_metrics(noisy_row, "high", ref_image, noisy_image)
            noise_rows.append(noisy_row)

            filtered_lookup = {result.name: result.image for result in filtered_results}
            for result in filtered_results:
                filtered_row: dict[str, float | str] = {
                    "image": name,
                    "noise_type": noise_type,
                    "stage": "filtered",
                    "filter_name": result.name,
                    "reference_available": "yes" if ref_image is not None else "no",
                }
                filtered_row.update(compute_basic_stats(result.image))
                add_target_metrics(filtered_row, "low", low_image, result.image)
                add_target_metrics(filtered_row, "high", ref_image, result.image)
                noise_rows.append(filtered_row)

                denoise_then = enhance_for_order(result.image)
                enhance_then = {
                    "Mean filter": mean_filter(enhance_for_order(noisy_image), size=3),
                    "Gaussian filter": gaussian_filter_rgb(enhance_for_order(noisy_image), sigma=1.0),
                    "Median filter": median_filter_rgb(enhance_for_order(noisy_image), size=3),
                }[result.name]

                for stage_name, stage_image in [
                    ("denoise_then_enhance", denoise_then),
                    ("enhance_then_denoise", enhance_then),
                ]:
                    stage_row: dict[str, float | str] = {
                        "image": name,
                        "noise_type": noise_type,
                        "stage": stage_name,
                        "filter_name": result.name,
                        "reference_available": "yes" if ref_image is not None else "no",
                    }
                    stage_row.update(compute_basic_stats(stage_image))
                    add_target_metrics(stage_row, "low", low_image, stage_image)
                    add_target_metrics(stage_row, "high", ref_image, stage_image)
                    noise_rows.append(stage_row)

            chosen_filter_name = best_order_filter_name(noise_type)
            chosen_filtered = filtered_lookup[chosen_filter_name]
            chosen_denoise_then = enhance_for_order(chosen_filtered)
            chosen_enhance_then = {
                "Mean filter": mean_filter(enhance_for_order(noisy_image), size=3),
                "Gaussian filter": gaussian_filter_rgb(enhance_for_order(noisy_image), sigma=1.0),
                "Median filter": median_filter_rgb(enhance_for_order(noisy_image), size=3),
            }[chosen_filter_name]
            make_noise_order_figure(
                name,
                noise_type.replace("_", " ").title(),
                low_image,
                noisy_image,
                chosen_filtered,
                chosen_denoise_then,
                chosen_enhance_then,
                ref_image,
                noise_order_dir / f"{name}_{noise_type}_order.png",
            )

        print(f"Processed {name}")

    write_csv(
        method_rows,
        OUTPUT_DIR / "metrics.csv",
        preferred=[
            "image",
            "method",
            "reference_available",
            "mean_brightness",
            "variance",
            "contrast_std",
            "entropy",
            "michelson_contrast",
            "edge_pixels",
            "edge_ratio",
            "mse_to_low",
            "psnr_to_low",
            "mse_to_high",
            "psnr_to_high",
            "edge_f1_to_low",
            "edge_f1_to_high",
        ],
    )
    write_csv(
        noise_rows,
        OUTPUT_DIR / "noise_metrics.csv",
        preferred=[
            "image",
            "noise_type",
            "stage",
            "filter_name",
            "reference_available",
            "mean_brightness",
            "variance",
            "contrast_std",
            "entropy",
            "mse_to_low",
            "psnr_to_low",
            "mse_to_high",
            "psnr_to_high",
            "edge_f1_to_low",
            "edge_f1_to_high",
        ],
    )
    write_csv(
        edge_rows,
        OUTPUT_DIR / "edge_metrics.csv",
        preferred=[
            "image",
            "method",
            "reference_available",
            "edge_pixels",
            "edge_ratio",
            "edge_precision_to_high",
            "edge_recall_to_high",
            "edge_f1_to_high",
            "edge_iou_to_high",
        ],
    )
    write_summary_markdown(method_rows, OUTPUT_DIR / "summary.md")
    write_coverage_markdown(OUTPUT_DIR / "coverage_check.md")
    print(f"Results written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
