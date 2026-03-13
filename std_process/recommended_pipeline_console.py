from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.io import fits
from astropy.stats import sigma_clip
from numpy.lib.stride_tricks import sliding_window_view


FITS_SUFFIXES = {".fits", ".fit", ".fts"}


@dataclass
class FitsImage:
    path: Path
    data: np.ndarray
    header: Any


def read_fits_image(path: str | Path) -> FitsImage:
    p = Path(path)
    with fits.open(p) as hdul:
        hdu = None
        for candidate in hdul:
            if getattr(candidate, "data", None) is None:
                continue
            if candidate.data is None:
                continue
            if isinstance(candidate.data, np.ndarray):
                hdu = candidate
                break
        if hdu is None:
            raise ValueError(f"No image data found in FITS: {p}")
        data = np.squeeze(np.asarray(hdu.data))
        header = hdu.header
    return FitsImage(path=p, data=data, header=header)


def write_fits_image(path: str | Path, data: np.ndarray, header: Any | None = None, overwrite: bool = True) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(data=np.asarray(data), header=header)
    fits.HDUList([hdu]).writeto(p, overwrite=overwrite)


def _fill_nan_with_median(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    out = arr.copy()
    m = np.isfinite(out)
    if not np.any(m):
        return np.zeros_like(out, dtype=np.float64)
    med = float(np.median(out[m]))
    out[~m] = med
    return out


def _upsample_bilinear(grid: np.ndarray, h: int, w: int) -> np.ndarray:
    gy, gx = grid.shape
    if gy == h and gx == w:
        return grid.astype(np.float64, copy=False)
    x_src = np.linspace(0.0, 1.0, gx)
    y_src = np.linspace(0.0, 1.0, gy)
    x_dst = np.linspace(0.0, 1.0, w)
    y_dst = np.linspace(0.0, 1.0, h)
    grid_x = np.empty((gy, w), dtype=np.float64)
    for i in range(gy):
        grid_x[i, :] = np.interp(x_dst, x_src, grid[i, :])
    out = np.empty((h, w), dtype=np.float64)
    for j in range(w):
        out[:, j] = np.interp(y_dst, y_src, grid_x[:, j])
    return out


def _max_filter1d(x: np.ndarray, k: int) -> np.ndarray:
    n = x.size
    r = max(1, int(k))
    if n == 0:
        return x.copy()
    pad = r // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty(n, dtype=np.float64)
    dq: deque[int] = deque()
    for i in range(xp.size):
        while dq and dq[0] <= i - r:
            dq.popleft()
        while dq and xp[dq[-1]] <= xp[i]:
            dq.pop()
        dq.append(i)
        if i >= r - 1 and (i - r + 1) < n:
            out[i - r + 1] = xp[dq[0]]
    return out


def _max_filter2d(img: np.ndarray, k: int) -> np.ndarray:
    tmp = np.empty_like(img, dtype=np.float64)
    out = np.empty_like(img, dtype=np.float64)
    for y in range(img.shape[0]):
        tmp[y, :] = _max_filter1d(img[y, :], k)
    for x in range(img.shape[1]):
        out[:, x] = _max_filter1d(tmp[:, x], k)
    return out


def _estimate_source_mask(data: np.ndarray, sigma: float = 3.5, dilate_radius: int = 2) -> np.ndarray:
    img = _fill_nan_with_median(data)
    clipped = sigma_clip(img, sigma=3.0, maxiters=3, masked=True)
    vals = np.asarray(clipped.compressed(), dtype=np.float64)
    if vals.size == 0:
        vals = img.reshape(-1)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + 1e-6
    robust_std = 1.4826 * mad
    src = img > (med + sigma * robust_std)
    if dilate_radius <= 0:
        return src
    return _max_filter2d(src.astype(np.float64), 2 * dilate_radius + 1) > 0.5


def estimate_background_mesh(data: np.ndarray, box: int = 64, clip_sigma: float = 3.0) -> np.ndarray:
    img = _fill_nan_with_median(data)
    h, w = img.shape
    b = max(8, int(box))
    source_mask = _estimate_source_mask(img, sigma=max(2.5, clip_sigma), dilate_radius=2)
    ys = list(range(0, h, b))
    xs = list(range(0, w, b))
    grid = np.zeros((len(ys), len(xs)), dtype=np.float64)
    for iy, y0 in enumerate(ys):
        y1 = min(h, y0 + b)
        for ix, x0 in enumerate(xs):
            x1 = min(w, x0 + b)
            patch = img[y0:y1, x0:x1]
            patch_mask = source_mask[y0:y1, x0:x1]
            patch_bg = patch[~patch_mask]
            if patch_bg.size < max(16, (patch.size // 10)):
                patch_bg = patch.reshape(-1)
            clipped = sigma_clip(patch_bg, sigma=clip_sigma, maxiters=3, masked=True)
            vals = np.asarray(clipped.compressed(), dtype=np.float64)
            if vals.size == 0:
                vals = patch_bg.reshape(-1)
            grid[iy, ix] = float(np.median(vals))
    return _upsample_bilinear(grid, h, w)


def process_recommended_pipeline(
    data: np.ndarray,
    box: int,
    clip_sigma: float,
    denoise_sigma: float,
    mix_alpha: float,
) -> np.ndarray:
    img = _fill_nan_with_median(data)
    bg = estimate_background_mesh(img, box=max(16, int(box)), clip_sigma=float(clip_sigma))
    resid = img - bg
    den = np.asarray(
        convolve_fft(
            resid,
            Gaussian2DKernel(max(0.5, float(denoise_sigma))),
            normalize_kernel=True,
            boundary="fill",
            fill_value=float(np.median(resid)),
        ),
        dtype=np.float64,
    )
    a = float(np.clip(mix_alpha, 0.0, 1.0))
    mix = a * resid + (1.0 - a) * den
    return mix


def _restore_nonfinite_mask(output: np.ndarray, original: np.ndarray) -> np.ndarray:
    out = np.asarray(output, dtype=np.float64).copy()
    orig = np.asarray(original)
    nonfinite = ~np.isfinite(orig)
    if np.any(nonfinite):
        out[nonfinite] = np.nan
    return out


def _median_filter_2d(data: np.ndarray, ksize: int = 3) -> np.ndarray:
    k = int(ksize)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    if k == 1:
        return np.asarray(data, dtype=np.float64)

    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        return arr

    pad = k // 2
    padded = np.pad(arr, ((pad, pad), (pad, pad)), mode="edge")
    windows = sliding_window_view(padded, (k, k))
    return np.median(windows, axis=(-2, -1))


def _collect_fits_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in FITS_SUFFIXES:
            files.append(p)
    files.sort()
    return files


def _output_name_for_index(base_name: str, index: int, total: int) -> str:
    p = Path(base_name)
    ext = p.suffix.lower()
    if ext not in FITS_SUFFIXES:
        ext = ".fits"
    stem = p.stem if p.stem else "output"
    if total <= 1:
        return f"{stem}{ext}"
    digits = max(3, len(str(total)))
    return f"{stem}_{index:0{digits}d}{ext}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="std_process 独立推荐流程控制台工具")
    parser.add_argument("-i", "--input-path", required=True, type=Path, help="输入路径（FITS 文件全路径或目录）")
    parser.add_argument("-o", "--output-name", type=str, default=None, help="输出文件名（示例: result.fits）")
    parser.add_argument("--box", type=int, default=48, help="mesh 分块大小，默认 48")
    parser.add_argument("--clip-sigma", type=float, default=3.0, help="sigma clip 阈值，默认 3.0")
    parser.add_argument("--median-ksize", type=int, default=3, help="中值滤波核大小，默认 3（默认开启，设为 1 关闭）")
    parser.add_argument("--denoise-sigma", type=float, default=2.0, help="高斯去噪 sigma，默认 2.0")
    parser.add_argument("--mix-alpha", type=float, default=0.7, help="细节混合比例，默认 0.7")
    parser.add_argument("--keep-negative", action="store_true", help="保留负值（默认会将负值裁剪为 0）")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在输出文件")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.input_path.exists():
        raise ValueError(f"输入路径不存在或不可访问: {args.input_path}")
    if args.input_path.is_file() and args.input_path.suffix.lower() not in FITS_SUFFIXES:
        raise ValueError("输入文件不是 FITS（仅支持 .fits/.fit/.fts）")
    if args.box < 16:
        raise ValueError("box 需要 >= 16")
    if args.median_ksize < 1:
        raise ValueError("median-ksize 需要 >= 1")
    if not (1.5 <= args.clip_sigma <= 8.0):
        raise ValueError("clip-sigma 需要在 [1.5, 8.0] 范围内")
    if not (0.5 <= args.denoise_sigma <= 20.0):
        raise ValueError("denoise-sigma 需要在 [0.5, 20.0] 范围内")
    if not (0.0 <= args.mix_alpha <= 1.0):
        raise ValueError("mix-alpha 需要在 [0.0, 1.0] 范围内")
def main() -> int:
    args = _build_parser().parse_args()
    try:
        _validate_args(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 2

    input_path = args.input_path.resolve()
    is_single_file = input_path.is_file()
    if args.output_name is None:
        base_dir = input_path.parent if is_single_file else input_path
        output_root = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        output_root = input_path.parent if is_single_file else input_path
    output_root.mkdir(parents=True, exist_ok=True)

    fits_files = [input_path] if is_single_file else _collect_fits_files(input_path)
    if not fits_files:
        print(f"[WARN] 未发现 FITS 文件: {input_path}")
        return 1

    print(f"[INFO] 输入路径: {input_path}")
    print(f"[INFO] 输出目录: {output_root}")
    if args.output_name is not None:
        print(f"[INFO] 输出文件名模板: {args.output_name}")
    print(f"[INFO] 文件总数: {len(fits_files)}")
    print(
        "[INFO] 参数:",
        {
            "box": args.box,
            "clip_sigma": args.clip_sigma,
            "median_ksize": args.median_ksize,
            "denoise_sigma": args.denoise_sigma,
            "mix_alpha": args.mix_alpha,
            "keep_negative": args.keep_negative,
        },
    )

    ok_count = 0
    fail_count = 0
    for idx, in_path in enumerate(fits_files, start=1):
        try:
            img = read_fits_image(in_path)
            arr = np.squeeze(img.data).astype(np.float64)
            if arr.ndim != 2:
                raise ValueError(f"仅支持 2D 图像，当前维度: {arr.shape}")
            arr_work = _median_filter_2d(arr, ksize=args.median_ksize)
            arr_work = _restore_nonfinite_mask(arr_work, arr)
            processed = process_recommended_pipeline(
                arr_work,
                box=args.box,
                clip_sigma=args.clip_sigma,
                denoise_sigma=args.denoise_sigma,
                mix_alpha=args.mix_alpha,
            )
            out_data = _restore_nonfinite_mask(processed, arr)
            if not args.keep_negative:
                out_data[np.isfinite(out_data) & (out_data < 0.0)] = 0.0
            if args.output_name is None:
                rel = Path(in_path.name) if is_single_file else in_path.relative_to(input_path)
                out_path = output_root / rel
            else:
                filename = _output_name_for_index(args.output_name, idx, len(fits_files))
                out_path = output_root / filename
            if out_path.exists() and not args.overwrite:
                raise FileExistsError(f"输出文件已存在: {out_path}（可加 --overwrite）")
            write_fits_image(out_path, out_data, header=img.header, overwrite=True)
            ok_count += 1
            print(f"[{idx}/{len(fits_files)}] OK   {out_path.name}")
        except Exception as exc:
            fail_count += 1
            print(f"[{idx}/{len(fits_files)}] FAIL {in_path} -> {exc}")

    print("-" * 60)
    print(f"[DONE] 成功: {ok_count} 失败: {fail_count} 总数: {len(fits_files)}")
    print(f"[DONE] 输出目录: {output_root}")
    return 0 if fail_count == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())

