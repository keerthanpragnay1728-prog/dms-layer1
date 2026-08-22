"""Our landmark detector: Haar finds the box, our trained model refines the
24 points inside it (milestone 6).

The pipeline is exactly the deployment path decided in milestone 3: largest
Haar face, the calibrated box_scale/box_shift_y mapping to the model's crop
box, grayscale crop resized to the model input, and the [0, 1] predictions
mapped back to frame coordinates through the same crop geometry the
training cache used.
"""

from __future__ import annotations

import numpy as np
import torch

from dms_layer1.config import require
from dms_layer1.data.crops import (CropBox, extract_square, square_box_around,
                                   to_frame_space)
from dms_layer1.detect.haar import HaarFaceDetector
from dms_layer1.detect.interface import LandmarkDetector, Landmarks24, as_gray
from dms_layer1.model.io import describe_weights, load_model, resolve_weights


class OurLandmarkDetector(LandmarkDetector):
    name = "ours"

    def __init__(self, cfg: dict, weights: str | None = None):
        self.haar = HaarFaceDetector(cfg)
        weights_path = resolve_weights(weights or require(cfg, "detector.weights"))
        self.model, self.meta = load_model(weights_path, cfg)
        self.model.eval()
        self.input_size = self.model.arch["input_size"]
        self.pixel_mean = float(require(cfg, "train.pixel_mean"))
        self.pixel_std = float(require(cfg, "train.pixel_std"))
        # Two-stage refinement: predict once from the calibrated Haar box,
        # rebuild a canonical box from those points, predict again. Worth
        # nothing when stage one is far outside the model's competent range
        # and worth several NME points once it is not, so it is measured per
        # calibration by scripts/diagnose_deploy_gap.py rather than assumed.
        self.refine_stages = int(require(cfg, "detector.refine_stages"))
        self.refine_expand = float(require(cfg, "detector.refine_expand"))
        if self.refine_stages < 1:
            raise ValueError("detector.refine_stages must be at least 1")
        print(describe_weights(weights_path, self.meta))

    def predict_in_box(self, gray: np.ndarray, box: CropBox) -> np.ndarray:
        """The model's 24 points for one crop box, in frame coordinates."""
        crop = extract_square(gray, box, self.input_size)
        x = (torch.from_numpy(np.ascontiguousarray(crop)).float() / 255.0
             - self.pixel_mean) / self.pixel_std
        with torch.no_grad():
            pred01 = self.model(x.unsqueeze(0).unsqueeze(0))[0].numpy()
        return to_frame_space(pred01.astype(np.float64), box)

    def detect(self, frame: np.ndarray) -> Landmarks24 | None:
        gray = as_gray(frame)
        box = self.haar.primary_crop_box(gray)
        if box is None:
            return None
        pts = self.predict_in_box(gray, box)
        for _ in range(self.refine_stages - 1):
            pts = self.predict_in_box(gray, square_box_around(pts, self.refine_expand))
        return Landmarks24(points=pts.astype(np.float32), source=self.name)
