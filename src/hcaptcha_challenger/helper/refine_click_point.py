from pathlib import Path
from typing import Union

import cv2
import numpy as np

from hcaptcha_challenger.models import PointCoordinate


def refine_click_point(
    image: Union[str, Path],
    point: PointCoordinate,
    *,
    window_radius: int = 80,
    min_component_area: int = 120,
) -> PointCoordinate:
    """
    Snap a predicted click point toward a safer interior point of the nearest local object.
    """
    img = cv2.imread(str(image))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image}")

    height, width = img.shape[:2]
    x = int(np.clip(point.x, 0, width - 1))
    y = int(np.clip(point.y, 0, height - 1))

    x0 = max(0, x - window_radius)
    y0 = max(0, y - window_radius)
    x1 = min(width, x + window_radius + 1)
    y1 = min(height, y + window_radius + 1)
    roi = img[y0:y1, x0:x1]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 40, 120)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.bitwise_or(edges, thresh)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    local_seed = (x - x0, y - y0)
    best_component = None
    best_score = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_component_area:
            continue

        filled = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(filled, [contour], -1, 255, thickness=-1)

        contains_seed = filled[local_seed[1], local_seed[0]] > 0
        distance = cv2.pointPolygonTest(contour, local_seed, measureDist=True)
        score = (0 if contains_seed else 1, -distance if contains_seed else abs(distance), -area)

        if best_score is None or score < best_score:
            best_score = score
            best_component = filled

    if best_component is None:
        return PointCoordinate(x=x, y=y)

    distance_map = cv2.distanceTransform(best_component, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(distance_map)
    refined_x = int(np.clip(x0 + max_loc[0], 0, width - 1))
    refined_y = int(np.clip(y0 + max_loc[1], 0, height - 1))

    return PointCoordinate(x=refined_x, y=refined_y)
