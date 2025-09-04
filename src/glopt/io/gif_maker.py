"""Lightweight GIF creation utilities.

Prefers Pillow if available; falls back to imageio. If neither is present,
raises a RuntimeError with a clear message. This module intentionally keeps a
small surface area to avoid heavy dependencies in the core flow.
"""

from __future__ import annotations

from collections.abc import Sequence


def write_gif_from_paths(image_paths: Sequence[str], out_path: str, duration_sec: float = 2.0, loop: int = 0) -> str:
    """Create an animated GIF from a sequence of image files.

    Args:
        image_paths: Ordered list of frame file paths (e.g., PNGs).
        out_path: Destination GIF path.
        duration_sec: Duration per frame in seconds.
        loop: Number of loops (0 = infinite).

    Returns:
        The output GIF path.
    """
    if not image_paths:
        raise ValueError("image_paths is empty")

    # Try Pillow first
    try:
        from PIL import Image

        frames = [Image.open(p).convert("RGBA") for p in image_paths]
        first, rest = frames[0], frames[1:]
        first.save(
            out_path,
            save_all=True,
            append_images=rest,
            duration=int(max(0.0, duration_sec) * 1000),
            loop=loop,
            disposal=2,
            optimize=False,
        )
        return out_path
    except Exception:
        pass

    # Fallback: imageio
    try:
        import imageio.v2 as imageio

        frames = [imageio.imread(p) for p in image_paths]
        imageio.mimsave(out_path, frames, duration=duration_sec, loop=loop)
        return out_path
    except Exception as e:  # pragma: no cover - depends on environment packages
        raise RuntimeError("GIF creation requires Pillow or imageio. Please install one of them (e.g., 'uv add pillow').") from e
