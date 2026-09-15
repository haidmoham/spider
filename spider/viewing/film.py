"""Render synchronized comparison films from recorded MuJoCo states."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import shutil
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..recording import STATE
from .stalk import apply_stalk_presentation, stalk_camera


def _matching_indices(recorded_times: np.ndarray, expected_times: np.ndarray) -> np.ndarray:
    """Find shared timestamps despite harmless floating-point drift."""
    relative = np.asarray(recorded_times, dtype=np.float64) - recorded_times[0]
    right = np.clip(np.searchsorted(relative, expected_times), 0, len(relative) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(
        np.abs(relative[left] - expected_times) <= np.abs(relative[right] - expected_times),
        left,
        right,
    )
    if not np.allclose(relative[indices], expected_times, rtol=0, atol=1e-9):
        raise ValueError("recording does not contain every requested comparison timestamp")
    return indices


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path("C:/Windows/Fonts/segoeuib.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def render_comparison(
    directories: list[str | Path],
    labels: list[str],
    output: str | Path,
    fps: int = 50,
    *,
    duration_s: float = 5.0,
    size: tuple[int, int] = (1280, 720),
) -> Path:
    """Render four recorded treatments on one clock without physics integration."""
    if len(directories) != 4 or len(labels) != 4:
        raise ValueError("A comparison film requires exactly four directories and labels")
    if fps <= 0 or duration_s <= 0:
        raise ValueError("fps and duration_s must be positive")
    width, height = size
    if width % 2 or height % 2:
        raise ValueError("Video dimensions must be even")

    frame_count = round(duration_s * fps)
    pane_width, pane_height = width // 2, height // 2
    models: list[mujoco.MjModel] = []
    data: list[mujoco.MjData] = []
    states: list[np.ndarray] = []
    expected_times = np.arange(frame_count, dtype=np.float64) / fps
    for raw_directory in directories:
        directory = Path(raw_directory)
        model = mujoco.MjModel.from_binary_path(str(directory / "model.mjb"))
        apply_stalk_presentation(model)
        models.append(model)
        data.append(mujoco.MjData(model))
        with np.load(directory / "states.npz", allow_pickle=False) as archive:
            recorded_states = archive["states"].copy()
            recorded_times = archive["times"].copy()
        try:
            indices = _matching_indices(recorded_times, expected_times)
        except ValueError as error:
            raise ValueError(
                f"{directory} does not contain every exact {fps} fps comparison timestamp"
            ) from error
        # Select recorded poses at shared timestamps. Never interpolate or hold a final frame.
        states.append(recorded_states[indices])

    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode the comparison film")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]

    label_font = _font(22)
    time_font = _font(16)
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        with ExitStack() as stack:
            renderers = [
                stack.enter_context(mujoco.Renderer(model, pane_height, pane_width))
                for model in models
            ]
            for frame_index in range(frame_count):
                canvas = Image.new("RGB", size, "#11141a")
                for pane, (model, datum, renderer, treatment_states, label) in enumerate(
                    zip(models, data, renderers, states, labels, strict=True)
                ):
                    mujoco.mj_setState(model, datum, treatment_states[frame_index], STATE)
                    # Rebuild display transforms only. The recorded trajectory is never advanced.
                    mujoco.mj_forward(model, datum)
                    renderer.update_scene(datum, camera=stalk_camera(model, datum))
                    image = Image.fromarray(renderer.render())
                    left = (pane % 2) * pane_width
                    top = (pane // 2) * pane_height
                    canvas.paste(image, (left, top))
                    draw = ImageDraw.Draw(canvas, "RGBA")
                    draw.rectangle((left, top, left + pane_width, top + 40), fill=(8, 10, 14, 205))
                    draw.text((left + 16, top + 8), label, font=label_font, fill=(240, 235, 225, 255))
                    draw.text(
                        (left + pane_width - 72, top + 11),
                        f"{expected_times[frame_index]:.2f}s",
                        font=time_font,
                        fill=(172, 183, 202, 255),
                    )
                if process.stdin is None:
                    raise RuntimeError("ffmpeg input pipe closed")
                process.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())
        if process.stdin is not None:
            process.stdin.close()
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")
    except BaseException:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        raise
    return destination
