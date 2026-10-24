"""Parse the raw ATD once into numpy arrays and cache them, so nothing else re-streams the text.

    from tools.atd_cache import load_arrays
    atd = load_arrays()                 # tactical; load_arrays("broadcast") for the other
    atd["players"]   # (n_frames, n_slots, 3) float32 -- x, y (normalised 0-1), speed m/s; NaN = missing
    atd["ball"]      # (n_frames, 3) float32 -- x, y, z or speed as the file gives it; NaN = missing
    atd["team"]      # (n_slots,) int8 -- 0 home, 1 away

Indexed by VIDEO frame (row i = submission row i): the ATD's frame_id - 1 is applied here once.

The cache lives in data/cache/, which is gitignored with the rest of data/ -- it is the match data
in another format and must never be committed or shared.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, DATA_DIR, DEFAULT_SOURCE, require, source_frames  # noqa: E402

# Params
CACHE_DIR = os.path.join(DATA_DIR, "cache")
HOME, AWAY = 0, 1


def team_of_each_track(meta_path):
    """0/1 per track slot; the first team in the metadata is home (see starter/baseline.py)."""
    root = ET.parse(meta_path).getroot()
    home_id = [team.get("id") for team in root.iter("Team")][0]
    return np.array([HOME if p.get("teamId") == home_id else AWAY for p in root.iter("Player")],
                    dtype=np.int8)


def _group(text, width):
    """'x,y,s' -> list of floats, NaN-padded to width. 'NaN' parses as nan natively."""
    values = [float(v) for v in text.split(",")[:width]]
    return values + [np.nan] * (width - len(values))


def parse(source=DEFAULT_SOURCE, game=GAME, data_dir=None):
    """Stream the raw ATD into arrays. A minute or two for the tactical file."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    team = team_of_each_track(meta_path)
    n_frames, n_slots = source_frames(source, game), len(team)

    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # frame_id counts from 1, video frames from 0
            for slot, group in enumerate(parts[1].split(";")[:n_slots]):
                if group[0] != "N":
                    players[frame, slot] = _group(group, 3)
            if parts[2] and parts[2][0] != "N":
                ball[frame] = _group(parts[2], 3)
    return {"players": players, "ball": ball, "team": team}


def load_arrays(source=DEFAULT_SOURCE, game=GAME, data_dir=None, rebuild=False):
    """Cached parse(). The first call builds data/cache/<game>_<source>_atd.npz."""
    path = os.path.join(CACHE_DIR, "{}_{}_atd.npz".format(game, source))
    if rebuild or not os.path.isfile(path):
        arrays = parse(source, game, data_dir)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(path, **arrays)
        return arrays
    with np.load(path) as cached:
        return {key: cached[key] for key in cached.files}


if __name__ == '__main__':
    for name in sys.argv[1:] or [DEFAULT_SOURCE]:
        arrays = load_arrays(name, rebuild=True)
        print("{}: players {}  ball {}  -> cached in {}".format(
            name, arrays["players"].shape, arrays["ball"].shape, CACHE_DIR))
