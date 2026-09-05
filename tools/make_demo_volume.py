#!/usr/bin/env python3
"""Regenerate the Auto Contrast demo volume.

Synthetic CT-like phantom: a noisy air background plus five nested shells whose
densities sit close together on purpose. A plain linear window squashes those
five shells into near-identical greys — which is exactly the case Auto Contrast
is meant to pull apart.

    python tools/make_demo_volume.py

Writes testdata/demo_shells_96.tif — a multipage uint16 TIFF. Import it with
File > Import Volume..., choose the TIFF filter, and accept the pre-filled
metadata dialog (96 x 96 x 96, uint16).

The generator lives in the repo; its output does not. testdata/ is gitignored,
so regenerate the volume rather than committing 1.8 MB of synthetic data.
"""
import os
import numpy as np

try:
    import tifffile
except ImportError:
    raise SystemExit('tifffile is required: pip install tifffile')

N = 96
SEED = 3          # the seed used for the before/after screenshots
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_REPO_ROOT, 'testdata')
OUT = os.path.join(OUT_DIR, f'demo_shells_{N}.tif')

# (radius, mean intensity, noise sigma) — the four inner shells are only
# ~2500 apart on a 0..65535 scale, so they are the hard case.
SHELLS = (
    (40, 21000, 700),
    (32, 23500, 700),
    (24, 26000, 700),
    (14, 28500, 700),
    (7, 52000, 1200),   # a dense core, far brighter than everything else
)


def build(n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    vol = rng.normal(400, 150, size=(n, n, n)).astype(np.float32)   # air
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    r = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2 + (zz - n / 2) ** 2)
    for radius, mean, sd in SHELLS:
        msk = r < radius
        vol[msk] = rng.normal(mean, sd, size=int(msk.sum()))
    return np.clip(vol, 0, 65535).astype(np.uint16)


if __name__ == '__main__':
    vol = build()
    os.makedirs(OUT_DIR, exist_ok=True)
    # Pages are Z, so tifffile reads (Z, Y, X) — what load_volume_tiff expects.
    tifffile.imwrite(OUT, vol, photometric='minisblack')
    print(f'wrote {OUT}')
    print(f'  shape {vol.shape} (Z, Y, X), dtype {vol.dtype}, '
          f'{os.path.getsize(OUT) / 1e6:.1f} MB')
    print(f'  intensity range {vol.min()}..{vol.max()}')
