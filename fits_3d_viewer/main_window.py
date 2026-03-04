from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve_fft
from numpy.lib.stride_tricks import sliding_window_view
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from fits_3d_viewer.background import (
    BG_METHOD_MESH,
    BG_METHOD_MORPH,
    BG_METHOD_ORIGINAL,
    BG_METHOD_PIPELINE,
    BG_METHOD_POLY2,
    BG_METHOD_RPCA,
    BG_METHOD_WAVELET,
    remove_background_with_params,
)
from fits_3d_viewer.canvas import ImageCanvas
from fits_3d_viewer.config import AppConfig
from fits_3d_viewer.file_browser import FileBrowser, TileGroup, discover_tiles
from fits_3d_viewer.fits_io import read_fits_image, to_uint8_view, write_fits_image
from fits_3d_viewer.view3d import Dual3DView


class MainWindow(QMainWindow):
    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.setWindowTitle("FITS 3D Viewer")
        self._cfg = cfg

        self._ref_path: Path | None = None
        self._aligned_path: Path | None = None
        self._current_tile: TileGroup | None = None
        self._raw_ref: np.ndarray | None = None
        self._raw_aligned: np.ndarray | None = None
        self._disp_ref: np.ndarray | None = None
        self._disp_aligned: np.ndarray | None = None

        self._canvas = ImageCanvas()
        self._canvas.set_mode("view3d")
        self._canvas.cursor_pixel.connect(self._on_cursor_pixel)
        self._canvas.view3d_click.connect(self._on_view3d_click)

        self._view3d = Dual3DView()
        self._view3d.patch_size_changed.connect(self._on_patch_size_changed)
        self._view3d.set_patch_size(self._cfg.patch_size)

        self._file_browser = FileBrowser()
        self._file_browser.tile_selected.connect(self._on_tile_selected)
        self._file_browser.setMinimumWidth(220)
        self._file_browser.setMaximumWidth(360)

        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(500)
        self._blink_timer.timeout.connect(self._on_blink_tick)
        self._blinking = False
        self._compare_original = False
        self._active_bg_method = self._cfg.bg_method

        self._build_ui()
        self._setup_shortcuts()
        self._set_active_bg_method(self._cfg.bg_method, save=False)

        if cfg.data_dir:
            self._file_browser.set_data_dir(cfg.data_dir)

    def _build_ui(self) -> None:
        tb = QToolBar("tools")
        tb.setMovable(False)
        self.addToolBar(tb)

        act_set_dir = QAction("📁 设置数据目录", self)
        act_set_dir.triggered.connect(self.set_data_dir_dialog)
        tb.addAction(act_set_dir)

        act_refresh = QAction("🔄 刷新", self)
        act_refresh.setShortcut(QKeySequence("F5"))
        act_refresh.triggered.connect(self._refresh_file_list)
        tb.addAction(act_refresh)

        self._act_compare_original = QAction("对比原图 (C)", self)
        self._act_compare_original.setCheckable(True)
        self._act_compare_original.setShortcut(QKeySequence("C"))
        self._act_compare_original.toggled.connect(self._on_compare_original_toggled)
        tb.addAction(self._act_compare_original)

        act_clip_negative = QAction("负值置零", self)
        act_clip_negative.triggered.connect(self._clip_negative_in_current_view)
        tb.addAction(act_clip_negative)

        act_gaussian_smooth = QAction("高斯平滑", self)
        act_gaussian_smooth.triggered.connect(self._gaussian_smooth_current_view)
        tb.addAction(act_gaussian_smooth)

        act_median_filter = QAction("中值滤波", self)
        act_median_filter.triggered.connect(self._median_filter_current_view)
        tb.addAction(act_median_filter)

        act_adaptive_hist_eq = QAction("自适应直方图均衡化", self)
        act_adaptive_hist_eq.triggered.connect(self._adaptive_hist_eq_current_view)
        tb.addAction(act_adaptive_hist_eq)

        act_gamma_correction = QAction("Gamma校正", self)
        act_gamma_correction.triggered.connect(self._gamma_correct_current_view)
        tb.addAction(act_gamma_correction)

        act_lift_dark_weak = QAction("暗弱点提升", self)
        act_lift_dark_weak.triggered.connect(self._lift_dark_weak_pixels_current_view)
        tb.addAction(act_lift_dark_weak)

        act_batch_export = QAction("批量处理导出", self)
        act_batch_export.triggered.connect(self._batch_process_and_export)
        tb.addAction(act_batch_export)

        self._image_name_label = QLabel("  显示: --")
        tb.addWidget(self._image_name_label)

        self._center_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._center_splitter.addWidget(self._canvas)
        self._center_splitter.addWidget(self._view3d)
        self._center_splitter.setStretchFactor(0, 3)
        self._center_splitter.setStretchFactor(1, 2)
        self._center_splitter.setSizes([900, 450])

        self._method_panel = self._build_method_panel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._file_browser)
        splitter.addWidget(self._center_splitter)
        splitter.addWidget(self._method_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([250, 1000, 340])
        self.setCentralWidget(splitter)

        self._status_coord = QLabel("x=-, y=-")
        self._status_value = QLabel("")
        self._status_info = QLabel("")
        self._status_file = QLabel("")
        self.statusBar().addWidget(self._status_coord, 0)
        self.statusBar().addWidget(self._status_value, 0)
        self.statusBar().addWidget(self._status_info, 1)
        self.statusBar().addPermanentWidget(self._status_file, 1)

    def _build_method_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(300)
        panel.setMaximumWidth(420)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        self._active_method_label = QLabel("当前方法: -")
        self._active_method_label.setStyleSheet("color: #4fc3f7; font-weight: bold;")
        lay.addWidget(self._active_method_label)

        # Original
        g0 = QGroupBox("原图")
        g0l = QHBoxLayout(g0)
        b0 = QPushButton("应用")
        b0.clicked.connect(lambda: self._set_active_bg_method(BG_METHOD_ORIGINAL))
        g0l.addWidget(b0)
        lay.addWidget(g0)

        # Mesh
        g1 = QGroupBox("1) 分块背景建模")
        g1l = QHBoxLayout(g1)
        self._mesh_box = QSpinBox()
        self._mesh_box.setRange(16, 512)
        self._mesh_box.setValue(self._cfg.mesh_box)
        self._mesh_clip = QDoubleSpinBox()
        self._mesh_clip.setRange(1.5, 8.0)
        self._mesh_clip.setSingleStep(0.1)
        self._mesh_clip.setValue(self._cfg.mesh_clip_sigma)
        b1 = QPushButton("应用")
        b1.clicked.connect(lambda: self._set_active_bg_method(BG_METHOD_MESH))
        g1l.addWidget(QLabel("box")); g1l.addWidget(self._mesh_box)
        g1l.addWidget(QLabel("sigma")); g1l.addWidget(self._mesh_clip)
        g1l.addWidget(b1)
        lay.addWidget(g1)

        # Morphology
        g2 = QGroupBox("2) 形态学背景")
        g2l = QHBoxLayout(g2)
        self._morph_radius = QSpinBox()
        self._morph_radius.setRange(2, 256)
        self._morph_radius.setValue(self._cfg.morph_radius)
        b2 = QPushButton("应用")
        b2.clicked.connect(lambda: self._set_active_bg_method(BG_METHOD_MORPH))
        g2l.addWidget(QLabel("radius")); g2l.addWidget(self._morph_radius); g2l.addWidget(b2)
        lay.addWidget(g2)

        # Poly
        g3 = QGroupBox("3) 多项式曲面拟合")
        g3l = QHBoxLayout(g3)
        self._poly_sample = QSpinBox()
        self._poly_sample.setRange(5000, 500000)
        self._poly_sample.setSingleStep(5000)
        self._poly_sample.setValue(self._cfg.poly_sample_size)
        self._poly_clip = QDoubleSpinBox()
        self._poly_clip.setRange(1.5, 8.0)
        self._poly_clip.setSingleStep(0.1)
        self._poly_clip.setValue(self._cfg.poly_clip_sigma)
        b3 = QPushButton("应用")
        b3.clicked.connect(lambda: self._set_active_bg_method(BG_METHOD_POLY2))
        g3l.addWidget(QLabel("sample")); g3l.addWidget(self._poly_sample)
        g3l.addWidget(QLabel("sigma")); g3l.addWidget(self._poly_clip)
        g3l.addWidget(b3)
        lay.addWidget(g3)

        # Wavelet
        g4 = QGroupBox("4) 小波多尺度")
        g4l = QHBoxLayout(g4)
        self._wav_sigma = QDoubleSpinBox()
        self._wav_sigma.setRange(2.0, 64.0)
        self._wav_sigma.setSingleStep(0.5)
        self._wav_sigma.setValue(self._cfg.wavelet_base_sigma)
        self._wav_levels = QSpinBox()
        self._wav_levels.setRange(2, 8)
        self._wav_levels.setValue(self._cfg.wavelet_levels)
        b4 = QPushButton("应用")
        b4.clicked.connect(lambda: self._set_active_bg_method(BG_METHOD_WAVELET))
        g4l.addWidget(QLabel("sigma")); g4l.addWidget(self._wav_sigma)
        g4l.addWidget(QLabel("levels")); g4l.addWidget(self._wav_levels)
        g4l.addWidget(b4)
        lay.addWidget(g4)

        # RPCA
        g5 = QGroupBox("5) 低秩+稀疏")
        g5l = QHBoxLayout(g5)
        self._rpca_rank = QSpinBox()
        self._rpca_rank.setRange(1, 10)
        self._rpca_rank.setValue(self._cfg.rpca_rank_keep)
        b5 = QPushButton("应用")
        b5.clicked.connect(lambda: self._set_active_bg_method(BG_METHOD_RPCA))
        g5l.addWidget(QLabel("rank")); g5l.addWidget(self._rpca_rank); g5l.addWidget(b5)
        lay.addWidget(g5)

        # Pipeline
        g6 = QGroupBox("6) 推荐流程")
        g6l = QVBoxLayout(g6)
        row1 = QHBoxLayout()
        self._pipe_box = QSpinBox(); self._pipe_box.setRange(16, 512); self._pipe_box.setValue(self._cfg.pipeline_box)
        self._pipe_clip = QDoubleSpinBox(); self._pipe_clip.setRange(1.5, 8.0); self._pipe_clip.setSingleStep(0.1); self._pipe_clip.setValue(self._cfg.pipeline_clip_sigma)
        row1.addWidget(QLabel("box")); row1.addWidget(self._pipe_box); row1.addWidget(QLabel("sigma")); row1.addWidget(self._pipe_clip)
        row2 = QHBoxLayout()
        self._pipe_denoise = QDoubleSpinBox(); self._pipe_denoise.setRange(0.5, 20.0); self._pipe_denoise.setSingleStep(0.1); self._pipe_denoise.setValue(self._cfg.pipeline_denoise_sigma)
        self._pipe_mix = QDoubleSpinBox(); self._pipe_mix.setRange(0.0, 1.0); self._pipe_mix.setSingleStep(0.05); self._pipe_mix.setValue(self._cfg.pipeline_mix_alpha)
        self._pipe_gain = QDoubleSpinBox(); self._pipe_gain.setRange(0.5, 40.0); self._pipe_gain.setSingleStep(0.5); self._pipe_gain.setValue(self._cfg.pipeline_asinh_gain)
        row2.addWidget(QLabel("denoise")); row2.addWidget(self._pipe_denoise)
        row2.addWidget(QLabel("mix")); row2.addWidget(self._pipe_mix)
        row2.addWidget(QLabel("gain")); row2.addWidget(self._pipe_gain)
        b6 = QPushButton("应用推荐流程")
        b6.clicked.connect(lambda: self._set_active_bg_method(BG_METHOD_PIPELINE))
        g6l.addLayout(row1); g6l.addLayout(row2); g6l.addWidget(b6)
        lay.addWidget(g6)

        lay.addStretch(1)
        return panel

    def _setup_shortcuts(self) -> None:
        sc_prev = QShortcut(QKeySequence("PgUp"), self)
        sc_prev.activated.connect(self._file_browser.go_prev)
        sc_next = QShortcut(QKeySequence("PgDown"), self)
        sc_next.activated.connect(self._file_browser.go_next)

    def _fits_filter(self) -> str:
        return "FITS (*.fits *.fit *.fts);;All (*.*)"

    def set_data_dir_dialog(self) -> None:
        start = self._cfg.data_dir or str(Path.cwd())
        p = QFileDialog.getExistingDirectory(self, "选择数据目录", start)
        if not p:
            return
        self._cfg.data_dir = str(Path(p))
        self._cfg.save()
        self._file_browser.set_data_dir(self._cfg.data_dir)
        self.statusBar().showMessage(f"数据目录: {self._cfg.data_dir}")

    def _refresh_file_list(self) -> None:
        if self._cfg.data_dir:
            self._file_browser.set_data_dir(self._cfg.data_dir)
            self.statusBar().showMessage("已刷新文件列表")

    def _on_patch_size_changed(self, size: int) -> None:
        self._cfg.patch_size = int(size)
        self._cfg.save()

    def _save_bg_params_to_cfg(self) -> None:
        self._cfg.mesh_box = int(self._mesh_box.value())
        self._cfg.mesh_clip_sigma = float(self._mesh_clip.value())
        self._cfg.morph_radius = int(self._morph_radius.value())
        self._cfg.poly_sample_size = int(self._poly_sample.value())
        self._cfg.poly_clip_sigma = float(self._poly_clip.value())
        self._cfg.wavelet_base_sigma = float(self._wav_sigma.value())
        self._cfg.wavelet_levels = int(self._wav_levels.value())
        self._cfg.rpca_rank_keep = int(self._rpca_rank.value())
        self._cfg.pipeline_box = int(self._pipe_box.value())
        self._cfg.pipeline_clip_sigma = float(self._pipe_clip.value())
        self._cfg.pipeline_denoise_sigma = float(self._pipe_denoise.value())
        self._cfg.pipeline_mix_alpha = float(self._pipe_mix.value())
        self._cfg.pipeline_asinh_gain = float(self._pipe_gain.value())
        self._cfg.bg_method = self._active_bg_method
        self._cfg.save()

    def _set_active_bg_method(self, method: str, save: bool = True) -> None:
        self._active_bg_method = method
        self._active_method_label.setText(f"当前方法: {method}")
        if save:
            self._save_bg_params_to_cfg()
        self._recompute_background_view()

    def _params_for_method(self, method: str) -> dict[str, float | int]:
        if method == BG_METHOD_MESH:
            return {"box": int(self._mesh_box.value()), "clip_sigma": float(self._mesh_clip.value())}
        if method == BG_METHOD_MORPH:
            return {"radius": int(self._morph_radius.value())}
        if method == BG_METHOD_POLY2:
            return {
                "sample_size": int(self._poly_sample.value()),
                "clip_sigma": float(self._poly_clip.value()),
            }
        if method == BG_METHOD_WAVELET:
            return {
                "base_sigma": float(self._wav_sigma.value()),
                "levels": int(self._wav_levels.value()),
            }
        if method == BG_METHOD_RPCA:
            return {"rank_keep": int(self._rpca_rank.value())}
        if method == BG_METHOD_PIPELINE:
            return {
                "box": int(self._pipe_box.value()),
                "clip_sigma": float(self._pipe_clip.value()),
                "denoise_sigma": float(self._pipe_denoise.value()),
                "mix_alpha": float(self._pipe_mix.value()),
                "asinh_gain": float(self._pipe_gain.value()),
            }
        return {}

    def _on_compare_original_toggled(self, on: bool) -> None:
        self._compare_original = bool(on)
        self._recompute_background_view()

    def _set_op_status(self, message: str) -> None:
        # 固定在状态栏标签显示，避免临时消息不易察觉或被覆盖。
        self._status_info.setText(message)
        self.statusBar().showMessage(message)
        print(f"[FITS3D] {message}", flush=True)

    def _clip_negative_in_current_view(self) -> None:
        showing_aligned = self._canvas.is_showing_aligned()
        target_name = "aligned" if showing_aligned else "reference"

        if showing_aligned:
            if self._disp_aligned is None:
                self.statusBar().showMessage("当前无 aligned 图像可处理")
                return
            target = self._disp_aligned
            slot = "b"
        else:
            if self._disp_ref is None:
                self.statusBar().showMessage("当前无 reference 图像可处理")
                return
            target = self._disp_ref
            slot = "a"

        neg_mask = np.isfinite(target) & (target < 0.0)
        neg_count = int(np.count_nonzero(neg_mask))
        if neg_count <= 0:
            self._set_op_status(f"{target_name} 图像中无负值像素")
            return

        clipped = np.array(target, copy=True)
        clipped[neg_mask] = 0.0

        if showing_aligned:
            self._disp_aligned = clipped
        else:
            self._disp_ref = clipped
        self._canvas.load_base_gray8(to_uint8_view(clipped), slot=slot)
        self._view3d.set_data(self._disp_ref, self._disp_aligned)
        self._set_op_status(f"{target_name} 负值置零完成: {neg_count} 个像素")

    def _gaussian_smooth_current_view(self) -> None:
        showing_aligned = self._canvas.is_showing_aligned()
        target_name = "aligned" if showing_aligned else "reference"
        sigma = 1.5

        if showing_aligned:
            if self._disp_aligned is None:
                self.statusBar().showMessage("当前无 aligned 图像可平滑")
                return
            target = self._disp_aligned
            slot = "b"
        else:
            if self._disp_ref is None:
                self.statusBar().showMessage("当前无 reference 图像可平滑")
                return
            target = self._disp_ref
            slot = "a"

        fill_val = float(np.nanmedian(target)) if np.isfinite(target).any() else 0.0
        smoothed = np.asarray(
            convolve_fft(
                target,
                Gaussian2DKernel(sigma),
                normalize_kernel=True,
                boundary="fill",
                fill_value=fill_val,
            ),
            dtype=np.float64,
        )

        if showing_aligned:
            self._disp_aligned = smoothed
        else:
            self._disp_ref = smoothed
        self._canvas.load_base_gray8(to_uint8_view(smoothed), slot=slot)
        self._view3d.set_data(self._disp_ref, self._disp_aligned)
        self._set_op_status(f"{target_name} 高斯平滑完成 (sigma={sigma})")

    def _median_filter_2d(self, data: np.ndarray, ksize: int = 3) -> np.ndarray:
        k = int(ksize)
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1
        if data.ndim != 2:
            return np.asarray(data, dtype=np.float64)

        arr = np.asarray(data, dtype=np.float64)
        if not np.isfinite(arr).all():
            fill_val = float(np.nanmedian(arr)) if np.isfinite(arr).any() else 0.0
            arr = np.array(arr, copy=True)
            arr[~np.isfinite(arr)] = fill_val

        pad = k // 2
        padded = np.pad(arr, ((pad, pad), (pad, pad)), mode="edge")
        windows = sliding_window_view(padded, (k, k))
        return np.nanmedian(windows, axis=(-2, -1))

    def _median_filter_current_view(self) -> None:
        showing_aligned = self._canvas.is_showing_aligned()
        target_name = "aligned" if showing_aligned else "reference"
        ksize = 3

        if showing_aligned:
            if self._disp_aligned is None:
                self.statusBar().showMessage("当前无 aligned 图像可滤波")
                return
            target = self._disp_aligned
            slot = "b"
        else:
            if self._disp_ref is None:
                self.statusBar().showMessage("当前无 reference 图像可滤波")
                return
            target = self._disp_ref
            slot = "a"

        filtered = self._median_filter_2d(target, ksize=ksize)
        if showing_aligned:
            self._disp_aligned = filtered
        else:
            self._disp_ref = filtered
        self._canvas.load_base_gray8(to_uint8_view(filtered), slot=slot)
        self._view3d.set_data(self._disp_ref, self._disp_aligned)
        self._set_op_status(f"{target_name} 中值滤波完成 (ksize={ksize})")

    def _adaptive_hist_eq_2d(self, data: np.ndarray, tile_size: int = 64, clip_limit: float = 0.02) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2:
            return arr
        finite = np.isfinite(arr)
        if not np.any(finite):
            return np.zeros_like(arr, dtype=np.float64)

        vals = arr[finite]
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.0))
        if hi <= lo:
            lo = float(np.min(vals))
            hi = float(np.max(vals))
            if hi <= lo:
                return np.zeros_like(arr, dtype=np.float64)

        norm = np.zeros_like(arr, dtype=np.float64)
        norm[finite] = np.clip((arr[finite] - lo) / (hi - lo), 0.0, 1.0)
        u8 = (norm * 255.0).astype(np.uint8)

        h, w = u8.shape
        ts = max(16, int(tile_size))
        out_u8 = np.zeros_like(u8, dtype=np.uint8)

        def _tile_lut(tile: np.ndarray) -> np.ndarray:
            hist = np.bincount(tile.ravel(), minlength=256).astype(np.float64)
            if clip_limit > 0.0:
                clip_val = max(1.0, float(clip_limit) * float(tile.size))
                excess = np.maximum(hist - clip_val, 0.0)
                hist = np.minimum(hist, clip_val)
                hist += excess.sum() / 256.0
            cdf = np.cumsum(hist)
            if cdf[-1] <= 0:
                return np.arange(256, dtype=np.uint8)
            cdf0 = cdf[0]
            den = max(cdf[-1] - cdf0, 1e-12)
            lut = np.floor((cdf - cdf0) / den * 255.0)
            return np.clip(lut, 0.0, 255.0).astype(np.uint8)

        for y0 in range(0, h, ts):
            y1 = min(h, y0 + ts)
            for x0 in range(0, w, ts):
                x1 = min(w, x0 + ts)
                tile = u8[y0:y1, x0:x1]
                lut = _tile_lut(tile)
                out_u8[y0:y1, x0:x1] = lut[tile]

        out = lo + (out_u8.astype(np.float64) / 255.0) * (hi - lo)
        out[~finite] = arr[~finite]
        return out

    def _adaptive_hist_eq_current_view(self) -> None:
        showing_aligned = self._canvas.is_showing_aligned()
        target_name = "aligned" if showing_aligned else "reference"
        tile_size = 64
        clip_limit = 0.02

        if showing_aligned:
            if self._disp_aligned is None:
                self.statusBar().showMessage("当前无 aligned 图像可均衡化")
                return
            target = self._disp_aligned
            slot = "b"
        else:
            if self._disp_ref is None:
                self.statusBar().showMessage("当前无 reference 图像可均衡化")
                return
            target = self._disp_ref
            slot = "a"

        eq = self._adaptive_hist_eq_2d(target, tile_size=tile_size, clip_limit=clip_limit)
        if showing_aligned:
            self._disp_aligned = eq
        else:
            self._disp_ref = eq
        self._canvas.load_base_gray8(to_uint8_view(eq), slot=slot)
        self._view3d.set_data(self._disp_ref, self._disp_aligned)
        self._set_op_status(
            f"{target_name} 自适应直方图均衡化完成 (tile={tile_size}, clip={clip_limit})"
        )

    def _gamma_correct_2d(self, data: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2:
            return arr
        g = max(0.05, float(gamma))
        finite = np.isfinite(arr)
        if not np.any(finite):
            return np.zeros_like(arr, dtype=np.float64)

        vals = arr[finite]
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.0))
        if hi <= lo:
            lo = float(np.min(vals))
            hi = float(np.max(vals))
            if hi <= lo:
                return arr.copy()

        out = arr.copy()
        x = np.clip((arr[finite] - lo) / (hi - lo), 0.0, 1.0)
        y = np.power(x, g)
        out[finite] = lo + y * (hi - lo)
        return out

    def _gamma_correct_current_view(self) -> None:
        showing_aligned = self._canvas.is_showing_aligned()
        target_name = "aligned" if showing_aligned else "reference"
        gamma = 1.2

        if showing_aligned:
            if self._disp_aligned is None:
                self.statusBar().showMessage("当前无 aligned 图像可Gamma校正")
                return
            target = self._disp_aligned
            slot = "b"
        else:
            if self._disp_ref is None:
                self.statusBar().showMessage("当前无 reference 图像可Gamma校正")
                return
            target = self._disp_ref
            slot = "a"

        corrected = self._gamma_correct_2d(target, gamma=gamma)
        if showing_aligned:
            self._disp_aligned = corrected
        else:
            self._disp_ref = corrected
        self._canvas.load_base_gray8(to_uint8_view(corrected), slot=slot)
        self._view3d.set_data(self._disp_ref, self._disp_aligned)
        self._set_op_status(f"{target_name} Gamma校正完成 (gamma={gamma})")

    def _lift_dark_weak_pixels(self, data: np.ndarray, ksize: int = 3, threshold_ratio: float = 0.30) -> tuple[np.ndarray, int, float]:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2:
            return arr, 0, 0.0

        finite = np.isfinite(arr)
        nonzero = finite & (arr > 0.0)
        if not np.any(nonzero):
            return arr.copy(), 0, 0.0

        # 按“非零像素亮度的30%”定义提升阈值。
        nz_vals = arr[nonzero]
        base = float(np.max(nz_vals))
        threshold = max(0.0, float(threshold_ratio) * base)
        if threshold <= 0.0:
            return arr.copy(), 0, 0.0

        med = self._median_filter_2d(arr, ksize=max(3, int(ksize)))
        weak_mask = nonzero & (arr < threshold) & (arr < 0.8 * med)

        out = arr.copy()
        count = int(np.count_nonzero(weak_mask))
        if count > 0:
            out[weak_mask] = threshold
        return out, count, threshold

    def _lift_dark_weak_pixels_current_view(self) -> None:
        showing_aligned = self._canvas.is_showing_aligned()
        target_name = "aligned" if showing_aligned else "reference"

        if showing_aligned:
            if self._disp_aligned is None:
                self.statusBar().showMessage("当前无 aligned 图像可处理")
                return
            target = self._disp_aligned
            slot = "b"
        else:
            if self._disp_ref is None:
                self.statusBar().showMessage("当前无 reference 图像可处理")
                return
            target = self._disp_ref
            slot = "a"

        lifted, count, threshold = self._lift_dark_weak_pixels(target, ksize=3, threshold_ratio=0.30)
        if showing_aligned:
            self._disp_aligned = lifted
        else:
            self._disp_ref = lifted
        self._canvas.load_base_gray8(to_uint8_view(lifted), slot=slot)
        self._view3d.set_data(self._disp_ref, self._disp_aligned)
        self._set_op_status(
            f"{target_name} 暗弱点提升完成 | 阈值: {threshold:.2f} (30%) | 提亮像素: {count} 个"
        )

    def _batch_process_and_export(self) -> None:
        if not self._cfg.data_dir:
            QMessageBox.warning(self, "批量处理导出", "请先设置数据目录。")
            return

        input_root = Path(self._cfg.data_dir)
        if not input_root.exists() or not input_root.is_dir():
            QMessageBox.warning(self, "批量处理导出", "数据目录不存在或不可访问。")
            return

        output_root = input_root / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root.mkdir(parents=True, exist_ok=True)

        tiles = discover_tiles(input_root)
        input_files: list[Path] = []
        for t in tiles:
            if t.reference is not None and t.reference.is_file():
                input_files.append(t.reference)
        if not input_files:
            QMessageBox.information(self, "批量处理导出", "未发现可处理的 FITS 文件。")
            return

        method = self._active_bg_method
        params = self._params_for_method(method)
        ok_count = 0
        fail_count = 0

        for idx, in_path in enumerate(input_files, start=1):
            try:
                img = read_fits_image(in_path)
                arr = np.squeeze(img.data).astype(np.float64)
                processed = remove_background_with_params(arr, method=method, params=params)
                clipped = np.array(processed, copy=True)
                neg_mask = np.isfinite(clipped) & (clipped < 0.0)
                clipped[neg_mask] = 0.0

                try:
                    rel = in_path.relative_to(input_root)
                except ValueError:
                    rel = Path(in_path.name)
                out_path = output_root / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                write_fits_image(out_path, clipped, header=img.header, overwrite=True)
                ok_count += 1
            except Exception:
                fail_count += 1
            self.statusBar().showMessage(
                f"批量处理中: {idx}/{len(input_files)} | 方法: {method} | 已完成: {ok_count} | 失败: {fail_count}"
            )

        QMessageBox.information(
            self,
            "批量处理导出完成",
            f"方法: {method}\n总文件: {len(input_files)}\n成功: {ok_count}\n失败: {fail_count}\n导出目录: {output_root}",
        )

    def _load_reference(self, path: Path) -> None:
        img = read_fits_image(path)
        self._raw_ref = np.squeeze(img.data).astype(np.float64)
        self._ref_path = path
        self._status_file.setText(f"📷 {path.name}")
        self._recompute_background_view()
        self._canvas.fit_view()

    def _load_aligned(self, path: Path) -> None:
        img = read_fits_image(path)
        self._raw_aligned = np.squeeze(img.data).astype(np.float64)
        self._aligned_path = path
        self._recompute_background_view()

    def _on_tile_selected(self, tile: TileGroup) -> None:
        self._current_tile = tile
        self._canvas.clear_all()
        self._ref_path = None
        self._aligned_path = None
        self._raw_ref = None
        self._raw_aligned = None
        self._disp_ref = None
        self._disp_aligned = None

        if tile.reference:
            self._load_reference(tile.reference)
        if tile.aligned:
            self._load_aligned(tile.aligned)

        self._image_name_label.setText("  显示: reference")
        self.setWindowTitle(f"FITS 3D Viewer - {tile.tile_id}")

    def _on_view3d_click(self, x: int, y: int) -> None:
        patch_size = self._view3d.get_patch_size()
        self._canvas.show_region_rect(x, y, patch_size)
        self._view3d.update_view(x, y)
        self.statusBar().showMessage(f"3D 查看: 中心({x}, {y})  {patch_size}×{patch_size} px")

    def _on_cursor_pixel(self, x: int, y: int, _code: int) -> None:
        self._status_coord.setText(f"x={x}, y={y}")
        val_str = ""
        if self._disp_ref is not None and 0 <= y < self._disp_ref.shape[0] and 0 <= x < self._disp_ref.shape[1]:
            val_str = f"ref={self._disp_ref[y, x]:.1f}"
        if self._disp_aligned is not None and 0 <= y < self._disp_aligned.shape[0] and 0 <= x < self._disp_aligned.shape[1]:
            if val_str:
                val_str += f"  ali={self._disp_aligned[y, x]:.1f}"
            else:
                val_str = f"ali={self._disp_aligned[y, x]:.1f}"
        self._status_value.setText(val_str)

    def _recompute_background_view(self) -> None:
        method = self._active_bg_method
        effective_method = BG_METHOD_ORIGINAL if self._compare_original else method
        params = self._params_for_method(effective_method)

        self._disp_ref = None
        self._disp_aligned = None

        if self._raw_ref is not None:
            self._disp_ref = remove_background_with_params(self._raw_ref, method=effective_method, params=params)
            self._canvas.load_base_gray8(to_uint8_view(self._disp_ref), slot="a")
        if self._raw_aligned is not None:
            self._disp_aligned = remove_background_with_params(
                self._raw_aligned, method=effective_method, params=params
            )
            self._canvas.load_base_gray8(to_uint8_view(self._disp_aligned), slot="b")

        self._view3d.set_data(self._disp_ref, self._disp_aligned)
        method_name = method
        if self._compare_original:
            self.statusBar().showMessage(f"对比原图中 (C 关闭)  | 当前方法: {method_name}")
        else:
            self.statusBar().showMessage(f"去背景方法: {method_name}")

    def _on_blink_tick(self) -> None:
        self._canvas.toggle_base_image()
        name = self._canvas.current_base_name()
        self._image_name_label.setText(f"  显示: {name}")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._cfg.window_width = int(self.width())
        self._cfg.window_height = int(self.height())
        self._cfg.save()
        self._blink_timer.stop()
        event.accept()
