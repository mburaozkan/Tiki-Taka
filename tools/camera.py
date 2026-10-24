"""Where is the tactical camera? Recovered from the per-frame homographies. Aggregate only.

    python tools/camera.py              # print the camera position and its spread
    from camera import camera_position  # (x, y, height) in metres, pitch coordinates

Each homography maps the pitch plane to the image. With square pixels and the principal point at
the image centre, the only unknown intrinsic is the focal length f, which follows from the two
planar-calibration constraints (the pitch's x and y axes are orthogonal and of equal scale); then
[r1 r2 t] = K^-1 H / lambda and the camera centre is -R^T t. A camera that pans but does not move
gives the same centre on every frame, which checks the method.

Used to see the ball the way our camera does: a ball at height z, mapped to the ground through the
homography, appears where the camera's line of sight hits the pitch, displaced away from the
camera by z / (camera height - z) of its horizontal distance (project_airborne).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES  # noqa: E402

# Params
WIDTH_PX, HEIGHT_PX = 1920, 1080
SAMPLE_EVERY = 500
MAX_REPROJ = 1.0


def centre_from_homography(H, length, width):
    """(x, y, z) of the camera in metres (pitch coordinates), or None if degenerate."""
    Hm = H @ np.diag([1.0 / length, 1.0 / width, 1.0])        # metres -> pixels
    cx, cy = WIDTH_PX / 2.0, HEIGHT_PX / 2.0
    Hc = Hm.copy()
    Hc[0] -= cx * Hm[2]
    Hc[1] -= cy * Hm[2]
    a, b = Hc[:, 0], Hc[:, 1]
    cands = []
    den = a[2] * b[2]
    if abs(den) > 1e-12:
        cands.append(-(a[0] * b[0] + a[1] * b[1]) / den)
    den = b[2] ** 2 - a[2] ** 2
    if abs(den) > 1e-12:
        cands.append(((a[0] ** 2 + a[1] ** 2) - (b[0] ** 2 + b[1] ** 2)) / den)
    f2 = [c for c in cands if c > 0]
    if not f2:
        return None
    f = np.sqrt(np.mean(f2))
    Kinv = np.diag([1.0 / f, 1.0 / f, 1.0])
    A = Kinv @ Hc
    lam = 1.0 / np.linalg.norm(A[:, 0])
    r1, r2, t = lam * A[:, 0], lam * A[:, 1], lam * A[:, 2]
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)
    c = -R.T @ t
    # The overall sign of [r1 r2 t] is not fixed by H; flipping it mirrors only the height
    # (r1 . t and r2 . t are unchanged), so the camera is (cx, cy, |cz|).
    return c[0], c[1], abs(c[2]), f


def camera_positions():
    from load_homography import load
    L, W = GAMES[GAME]["pitch"]
    H, err, valid = load()
    out = []
    for i in range(0, len(H), SAMPLE_EVERY):
        if valid[i] and err[i] <= MAX_REPROJ:
            r = centre_from_homography(H[i], L, W)
            if r is not None:
                out.append(r)
    return np.array(out)


def camera_position():
    """Median (x, y, height) in metres over the sampled frames."""
    return tuple(np.median(camera_positions()[:, :3], axis=0))


def project_airborne(x, y, z, cam):
    """Where a ball at (x, y, z) appears on the ground plane seen from cam (x, y, h), metres."""
    cx, cy, h = cam
    k = h / np.maximum(h - np.nan_to_num(z), 0.5)
    return cx + (x - cx) * k, cy + (y - cy) * k


def main():
    c = camera_positions()
    L, W = GAMES[GAME]["pitch"]
    print("frames used: {}".format(len(c)))
    for name, col in (("x (m, along the pitch)", 0), ("y (m, across)", 1), ("height (m)", 2),
                      ("focal length (px)", 3)):
        print("  {:<24} median {:8.1f}   p10 {:8.1f}   p90 {:8.1f}".format(
            name, *np.percentile(c[:, col], [50, 10, 90])))
    print("  pitch is {:.1f} x {:.1f} m; y = 0 and y = {:.0f} are the touchlines".format(L, W, W))


if __name__ == '__main__':
    main()
