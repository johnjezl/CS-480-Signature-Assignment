import cv2
import numpy as np
import itertools
from google.colab.patches import cv2_imshow

# Load your uploaded cube image
img = cv2.imread("Test.png")
orig = img.copy()

# Resize for stable processing
scale = 800 / img.shape[1]
img = cv2.resize(img, (0,0), fx=scale, fy=scale)
orig = cv2.resize(orig, (0,0), fx=scale, fy=scale)

import math
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

def dedup_boxes(boxes, min_dist=20):
    """Remove near-duplicate boxes based on center distance."""
    kept = []
    for (x, y, w, h, area) in sorted(boxes, key=lambda b: b[4], reverse=True):
        cx = x + w / 2
        cy = y + h / 2
        if all(math.hypot(cx - (kx + kw/2), cy - (ky + kh/2)) > min_dist
               for (kx, ky, kw, kh, ka) in kept):
            kept.append((x, y, w, h, area))
    return kept

def choose_nine(boxes):
    """From >9 boxes, keep the 9 closest to their centroid."""
    centers = [(x + w/2, y + h/2) for x, y, w, h, a in boxes]
    cx = sum(c[0] for c in centers) / len(centers)
    cy = sum(c[1] for c in centers) / len(centers)
    dists = [math.hypot(c[0] - cx, c[1] - cy) for c in centers]
    idx_sorted = sorted(range(len(boxes)), key=lambda i: dists[i])
    return [boxes[i] for i in idx_sorted[:9]]

def detect_grid(image):
    # resize for more stable parameters
    h, w = image.shape[:2]
    scale = 800.0 / w
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    work = image.copy()

    # Use brightness channel for stickers (they’re bright vs black plastic)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    hch, s, v = cv2.split(hsv)

    # Otsu threshold on V
    _, thresh = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        # size range tuned to your 2048x1536 pictures after resize
        if 5000 < area < 50000:
            x, y, w, h = cv2.boundingRect(c)
            # roughly square
            if 0.6 < w / float(h) < 1.4:
                boxes.append((x, y, w, h, area))

    # remove duplicates (inner/outer borders of the same sticker)
    boxes = dedup_boxes(boxes, min_dist=20)

    if len(boxes) < 9:
        print(f"Not enough squares: {len(boxes)}")
        return work, None

    if len(boxes) > 9:
        boxes = choose_nine(boxes)

    # --- sort into 3 rows using k-means on the y centers ---
    centers = [(x + w/2, y + h/2) for x, y, w, h, a in boxes]
    ys = np.float32([c[1] for c in centers]).reshape(-1, 1)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100, 0.2)
    K = 3
    _, labels, centers_y = cv2.kmeans(
        ys, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()
    centers_y = centers_y.flatten()
    row_order = np.argsort(centers_y)  # top to bottom

    ordered_boxes = []
    for row_idx in row_order:
        row_boxes = [b for b, l in zip(boxes, labels) if l == row_idx]
        # left to right
        row_boxes.sort(key=lambda b: b[0] + b[2]/2)
        ordered_boxes.extend(row_boxes)

    # in case of any weirdness, trim to 9
    ordered_boxes = ordered_boxes[:9]

    # extract mean color (BGR) and draw rectangles
    colors = []
    for (x, y, w, h, a) in ordered_boxes:
        roi = work[y:y+h, x:x+w]
        color = np.array(cv2.mean(roi))[:3].astype(int)
        colors.append(color)
        cv2.rectangle(work, (x, y), (x + w, y + h), (0, 255, 0), 3)

    grid = np.array(colors)  # shape (9, 3)
    return work, grid

def classifiy_grid(grid):
    if grid is None or len(grid) != 9:
        return "", None

    # grid is already (9,3) = B,G,R
    prediction = loaded_model.predict(grid[:, :3])

    mapping = {0:"F", 1:"U", 2:"R", 3:"L", 4:"B", 5:"D"}
    face = "".join(mapping[i] for i in prediction)
    return face, prediction

proc, grid = detect_grid(img)

processed, grid = detect_grid(img)
cv2_imshow(processed)

print("Grid:", grid)

if grid is not None:
    face, pred = classifiy_grid(grid)
    print("Predicted face string:", face)
    print("Raw model output:", pred)
