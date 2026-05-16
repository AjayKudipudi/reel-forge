"""Generate fixture inputs for local development."""
from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    photo = Image.fromarray(np.full((600, 800, 3), 200, dtype=np.uint8))
    photo.save(OUT / "jane.png")
    frames = np.tile(np.full((480, 640, 3), 100, dtype=np.uint8)[None], (24, 1, 1, 1))
    iio.imwrite(OUT / "sample.mp4", frames, fps=24, codec="libx264")
    print("regenerated:", OUT)


if __name__ == "__main__":
    main()
