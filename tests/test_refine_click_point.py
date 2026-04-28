from pathlib import Path

import cv2
import numpy as np

from hcaptcha_challenger.helper import refine_click_point
from hcaptcha_challenger.models import PointCoordinate


def test_refine_click_point_snaps_toward_shape_interior(tmp_path: Path):
    image = np.full((120, 120, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (30, 25), (90, 95), (30, 30, 30), thickness=-1)

    path = tmp_path / "shape.png"
    cv2.imwrite(str(path), image)

    refined = refine_click_point(path, PointCoordinate(x=32, y=30), window_radius=40)

    assert 45 <= refined.x <= 75
    assert 40 <= refined.y <= 80
