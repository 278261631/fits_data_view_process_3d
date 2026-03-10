from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fits_3d_viewer.background import BG_METHOD_PIPELINE, remove_background_with_params
from fits_3d_viewer.fits_io import read_fits_image, write_fits_image


FITS_SUFFIXES = {".fits", ".fit", ".fts"}


def _collect_fits_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in FITS_SUFFIXES:
            files.append(p)
    files.sort()
    return files


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FITS 推荐流程批量处理（robust_pipeline）控制台工具"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        type=Path,
        help="输入目录（递归搜索 *.fits/*.fit/*.fts）",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录（默认: 输入目录下 timestamp 子目录）",
    )
    parser.add_argument("--box", type=int, default=48, help="mesh 分块大小，默认 48")
    parser.add_argument("--clip-sigma", type=float, default=3.0, help="sigma clip 阈值，默认 3.0")
    parser.add_argument("--denoise-sigma", type=float, default=2.0, help="高斯去噪 sigma，默认 2.0")
    parser.add_argument("--mix-alpha", type=float, default=0.7, help="细节混合比例，默认 0.7")
    parser.add_argument("--asinh-gain", type=float, default=8.0, help="asinh 增益，默认 8.0")
    parser.add_argument(
        "--keep-negative",
        action="store_true",
        help="保留负值（默认会把负值裁剪为 0）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在输出文件（默认不覆盖）",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise ValueError(f"输入目录不存在或不可访问: {args.input_dir}")
    if args.box < 16:
        raise ValueError("box 需要 >= 16")
    if not (1.5 <= args.clip_sigma <= 8.0):
        raise ValueError("clip-sigma 需要在 [1.5, 8.0] 范围内")
    if not (0.5 <= args.denoise_sigma <= 20.0):
        raise ValueError("denoise-sigma 需要在 [0.5, 20.0] 范围内")
    if not (0.0 <= args.mix_alpha <= 1.0):
        raise ValueError("mix-alpha 需要在 [0.0, 1.0] 范围内")
    if not (0.5 <= args.asinh_gain <= 40.0):
        raise ValueError("asinh-gain 需要在 [0.5, 40.0] 范围内")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        _validate_args(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 2

    input_root = args.input_dir.resolve()
    if args.output_dir is None:
        output_root = input_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    fits_files = _collect_fits_files(input_root)
    if not fits_files:
        print(f"[WARN] 未发现 FITS 文件: {input_root}")
        return 1

    params: dict[str, float | int] = {
        "box": int(args.box),
        "clip_sigma": float(args.clip_sigma),
        "denoise_sigma": float(args.denoise_sigma),
        "mix_alpha": float(args.mix_alpha),
        "asinh_gain": float(args.asinh_gain),
    }

    ok_count = 0
    fail_count = 0
    print(f"[INFO] 输入目录: {input_root}")
    print(f"[INFO] 输出目录: {output_root}")
    print(f"[INFO] 文件总数: {len(fits_files)}")
    print(f"[INFO] 参数: {params}")

    for idx, in_path in enumerate(fits_files, start=1):
        try:
            img = read_fits_image(in_path)
            arr = np.squeeze(img.data).astype(np.float64)
            processed = remove_background_with_params(arr, method=BG_METHOD_PIPELINE, params=params)

            if args.keep_negative:
                out_data = np.asarray(processed, dtype=np.float64)
            else:
                out_data = np.asarray(processed, dtype=np.float64)
                out_data[np.isfinite(out_data) & (out_data < 0.0)] = 0.0

            rel = in_path.relative_to(input_root)
            out_path = output_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and not args.overwrite:
                raise FileExistsError(f"输出文件已存在: {out_path}（可加 --overwrite）")
            write_fits_image(out_path, out_data, header=img.header, overwrite=True)
            ok_count += 1
            print(f"[{idx}/{len(fits_files)}] OK   {rel}")
        except Exception as exc:
            fail_count += 1
            print(f"[{idx}/{len(fits_files)}] FAIL {in_path} -> {exc}")

    print("-" * 60)
    print(f"[DONE] 成功: {ok_count} 失败: {fail_count} 总数: {len(fits_files)}")
    print(f"[DONE] 输出目录: {output_root}")
    return 0 if fail_count == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())

