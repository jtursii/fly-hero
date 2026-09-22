"""Convert the halftone fly logo for the site's dark theme.

The source is a black halftone fly whose white background is already
transparent, but whose ink pixels still carry a full 0-255 grey ramp: the
halftone is encoded *twice*, once in alpha at the outline and once in
luminance inside it. Recoloring the RGB alone would leave the white dots
opaque white, so this collapses both into one coverage channel --

    coverage = (1 - luminance / 255) * (alpha / 255)

-- and paints that coverage in a single light tone. Black ink becomes fully
opaque light, white becomes fully transparent, and every intermediate dot
keeps its exact weight, so the halftone texture survives unchanged.

The result is trimmed to the ink's bounding box (the source has dead margin
on three sides) and kept at full resolution: it is displayed at 8-12% of the
CRT's width, so the browser downscales it, and a pre-shrunk asset would
alias the dots into moire on a high-DPR screen.

Usage:
    uv run python scripts/convert_logo.py \
        web/public/flylogo.png web/public/brand/flylogo.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

#: Pale cyan-white. Monochrome on purpose -- the logo sits over a highway of
#: saturated lane colours and must not read as one of them.
INK = (214, 236, 255)
#: Coverage below this is treated as bare background when trimming.
TRIM_EPS = 3


def convert(src: Path, dst: Path, ink: tuple[int, int, int] = INK) -> tuple[int, int]:
    """Recolor `src` onto transparency in `ink` and write it to `dst`.

    Returns the written image's (width, height).
    """
    a = np.array(Image.open(src).convert("RGBA"), dtype=np.float64)
    lum = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    coverage = (1.0 - lum / 255.0) * (a[..., 3] / 255.0)  # [H, W] in 0..1

    out = np.zeros(a.shape, dtype=np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = ink
    out[..., 3] = np.round(coverage * 255.0).astype(np.uint8)

    ys, xs = np.nonzero(out[..., 3] > TRIM_EPS)
    if len(ys) == 0:
        raise SystemExit(f"{src}: no ink found -- is it already converted?")
    out = out[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="RGBA").save(dst, optimize=True)
    return out.shape[1], out.shape[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    args = p.parse_args()
    w, h = convert(args.src, args.dst)
    print(f"{args.src} -> {args.dst}  {w}x{h}  {args.dst.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
