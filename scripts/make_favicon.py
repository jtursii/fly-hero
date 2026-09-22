"""Build the site's favicon from the fly logo.

The input is the already-converted watermark (`scripts/convert_logo.py`
output): light ink on transparency. That is exactly right inside the CRT and
exactly wrong in a browser tab, because a tab bar can be light or dark and a
light-ink-on-nothing mark is invisible on half of them. So the favicon gets
an opaque tile in the site's own panel colour, which reads on both.

Two other things a favicon needs that the watermark does not:

  * **Downsampling, once, from full resolution.** The source is a halftone.
    Letting the browser shrink a 774x604 halftone to 16 px aliases the dots
    into moire, so each size is resampled here with LANCZOS instead.
  * **Margin.** The watermark is trimmed hard to the ink's bounding box; a
    favicon that touches its own edges looks cramped next to every other
    tab, so the fly is inset into the tile.

Usage:
    uv run python scripts/make_favicon.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

SRC = Path("web/public/brand/flylogo.png")
OUT_DIR = Path("web/public/brand")

#: The site's panel colour (`--bg-panel`), so the tab reads as part of it.
TILE = (10, 14, 22, 255)
#: Fraction of the tile left as margin on each side.
INSET = 0.10
#: Corner radius as a fraction of the tile, matching the site's 8px-on-~64px
#: panel corners rather than a full squircle.
RADIUS = 0.18
#: Rendered at 4x and downsampled, so the rounded corners are antialiased --
#: `rounded_rectangle` itself draws them hard.
SS = 4

SIZES = {
    "favicon-32.png": 32,
    "favicon-180.png": 180,
    "favicon-512.png": 512,
}


def build(fly: Image.Image, size: int) -> Image.Image:
    """One square favicon tile at `size` px, antialiased."""
    big = size * SS
    tile = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    ImageDraw.Draw(tile).rounded_rectangle(
        (0, 0, big - 1, big - 1), radius=int(big * RADIUS), fill=TILE
    )

    # Fit the fly into the inset box, keeping its aspect ratio.
    box = int(big * (1 - 2 * INSET))
    scale = min(box / fly.width, box / fly.height)
    w, h = max(1, round(fly.width * scale)), max(1, round(fly.height * scale))
    resized = fly.resize((w, h), Image.LANCZOS)
    tile.alpha_composite(resized, ((big - w) // 2, (big - h) // 2))

    return tile.resize((size, size), Image.LANCZOS)


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC} -- run scripts/convert_logo.py first")
    fly = Image.open(SRC).convert("RGBA")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, size in SIZES.items():
        out = OUT_DIR / name
        build(fly, size).save(out, optimize=True)
        print(f"{out}  {size}x{size}  {out.stat().st_size:,} B")


if __name__ == "__main__":
    main()
