"""Render four synchronized saved C-1N checkpoint replays."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from render_walk_demo import add_reference_grid, font
from render_walk_speed_demo import Replay, load_replay
from spider.recording import STATE
from spider.viewing.googly import apply_googly_frame, apply_life_lights
from spider.viewing.stalk import stalk_camera


WIDTH, HEIGHT, FPS = 1080, 1350, 50
PANE_WIDTH, PANE_HEIGHT = WIDTH // 2, 426
STAGE_TOP, SHOT_SECONDS = 278, 5.0
CAMERAS = ((225.0, -25.0, 2.08), (270.0, -18.0, 2.08))
SOURCES = (
    (
        ROOT / "telemetry/tuning/20260915-forward-exploration-round-01/accepted_baseline/evaluation/mean-201",
        "notebook baseline", 0.37882745001803586,
        ROOT / "artifacts/ppo-crude-baseline-20260915/replays.zip", "3",
    ),
    (ROOT / "artifacts/walk_stable_100/replay", "walk_stable_100", 0.31750619506970296, None, None),
    (
        ROOT / "telemetry/tuning/20260915-policy-cadence-continuation-round-02/comparison-n00250/n00250/mean-201",
        "walk_fast_250", 0.7019195363016517,
        ROOT / "artifacts/walk_fast_200/experiments/policy-cadence-continuation-20260915/evidence.zip",
        "comparison-n00250/n00250/mean-201",
    ),
    (
        ROOT / "telemetry/tuning/20260915-policy-cadence-continuation-round-03/comparison-n00500/n00500/mean-201",
        "walk_fast_500", 0.8879874507126051,
        ROOT / "artifacts/walk_fast_500/experiments/cuda-continuation-20260915/evidence.zip",
        "comparison-n00500/n00500/mean-201",
    ),
)


def materialize_source(
    preferred: Path, archive_path: Path | None, entry: str | None, stack: ExitStack,
) -> Path:
    if all((preferred / name).is_file() for name in ("model.mjb", "states.npz")):
        return preferred
    if archive_path is None or entry is None:
        raise FileNotFoundError(f"Recorded replay is unavailable: {preferred}")
    temporary = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="c1n-demo-replay-")))
    with zipfile.ZipFile(archive_path) as archive:
        for name in ("model.mjb", "states.npz"):
            archive.extract(f"{entry}/{name}", temporary)
    return temporary / entry


def camera_for(replay: Replay, shot: int) -> mujoco.MjvCamera:
    camera = stalk_camera(replay.model, replay.data)
    camera.azimuth, camera.elevation, camera.distance = CAMERAS[shot]
    camera.lookat[:] = (float(replay.data.qpos[0]), float(replay.data.qpos[1]), 0.28)
    return camera


def draw_overlay(
    canvas: Image.Image, shot: int, time_s: float, replays: list[Replay], displacements: list[float],
) -> None:
    fade_layer = Image.new("RGBA", canvas.size)
    fade = ImageDraw.Draw(fade_layer)
    for y in range(300):
        fade.line((0, y, WIDTH, y), fill=(18, 18, 28, int(220 * (1 - y / 300))))
    for y in range(1130, HEIGHT):
        fade.line((0, y, WIDTH, y), fill=(18, 18, 28, int(220 * (y - 1130) / 220)))
    canvas.alpha_composite(fade_layer)
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((54, 45), "C-1N / four measured checkpoints", font=font(27), fill="#b7b1c3")
    draw.text((50, 94), "walking comparison", font=font(64, bold=True), fill="#eee6db")
    draw.text((54, 185), "saved mean replays · same scale · 1×", font=font(34), fill="#eee6db")
    draw.line((54, 253, 134, 253), fill="#f59470", width=3)

    for pane, (replay, displacement) in enumerate(zip(replays, displacements, strict=True)):
        column, row = pane % 2, pane // 2
        left, top = column * PANE_WIDTH, STAGE_TOP + row * PANE_HEIGHT
        draw.rectangle((left, top, left + PANE_WIDTH, top + 63), fill=(18, 18, 28, 178))
        draw.text((left + 24, top + 7), replay.name, font=font(22, bold=True), fill="#eee6db")
        draw.text(
            (left + 24, top + 35), f"measured {replay.speed:.3f} m/s",
            font=font(20), fill="#b7b1c3",
        )
        draw.rectangle(
            (left, top + PANE_HEIGHT - 43, left + PANE_WIDTH, top + PANE_HEIGHT),
            fill=(18, 18, 28, 158),
        )
        draw.text(
            (left + 24, top + PANE_HEIGHT - 36), f"Δx {displacement:.2f} m",
            font=font(22), fill="#eee6db",
        )
    draw.line((PANE_WIDTH, STAGE_TOP, PANE_WIDTH, 1130), fill=(183, 177, 195, 105), width=1)
    draw.line((0, STAGE_TOP + PANE_HEIGHT, WIDTH, STAGE_TOP + PANE_HEIGHT), fill=(183, 177, 195, 105), width=1)

    label = "three-quarter comparison" if shot == 0 else "side comparison · same replays"
    draw.text((54, 1190), label, font=font(34), fill="#eee6db")
    draw.text((54, 1254), "four recorded trajectories · normal 1× playback", font=font(27), fill="#b7b1c3")
    draw.text((1026, 1254), f"{time_s:.2f} s", anchor="ra", font=font(27), fill="#b7b1c3")


def render(destination: Path) -> Path:
    expected = np.arange(round(SHOT_SECONDS * FPS), dtype=np.float64) / FPS
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt",
        "yuv420p", "-movflags", "+faststart", str(destination),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        with ExitStack() as stack:
            resolved = [
                (materialize_source(path, archive, entry, stack), name, speed)
                for path, name, speed, archive, entry in SOURCES
            ]
            replays = [load_replay(path, name, speed, expected) for path, name, speed in resolved]
            renderers = [
                stack.enter_context(mujoco.Renderer(r.model, height=PANE_HEIGHT, width=PANE_WIDTH))
                for r in replays
            ]
            for shot in range(2):
                for index, time_s in enumerate(expected):
                    global_frame = shot * len(expected) + index
                    if global_frame % 100 == 0:
                        print(f"rendered {global_frame}/500", flush=True)
                    canvas = Image.new("RGBA", (WIDTH, HEIGHT), (18, 18, 28, 255))
                    displacements = []
                    for pane, (replay, renderer) in enumerate(zip(replays, renderers, strict=True)):
                        mujoco.mj_setState(replay.model, replay.data, replay.states[index], STATE)
                        apply_googly_frame(replay.model, replay.googly, index)
                        apply_life_lights(replay.model, float(time_s))
                        mujoco.mj_forward(replay.model, replay.data)
                        renderer.update_scene(replay.data, camera=camera_for(replay, shot))
                        add_reference_grid(renderer, extent=10.0)
                        left = (pane % 2) * PANE_WIDTH
                        top = STAGE_TOP + (pane // 2) * PANE_HEIGHT
                        canvas.paste(Image.fromarray(renderer.render()), (left, top))
                        displacements.append(float(replay.data.qpos[0]) - replay.origin_x)
                    draw_overlay(canvas, shot, float(time_s), replays, displacements)
                    if process.stdin is None:
                        raise RuntimeError("ffmpeg input pipe closed")
                    process.stdin.write(np.asarray(canvas.convert("RGB"), dtype=np.uint8).tobytes())
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
    print("rendered 500/500", flush=True)
    return destination.resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path.home() / "projects/demos/spider-walk-four-checkpoint-comparison.mp4",
    )
    print(render(parser.parse_args().output))


if __name__ == "__main__":
    main()
