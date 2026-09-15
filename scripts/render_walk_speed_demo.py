"""Render synchronized walk_stable_100 and cadence-action n=200 replays."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from render_walk_demo import add_reference_grid, font
from spider.recording import STATE
from spider.viewing.film import _matching_indices
from spider.viewing.googly import apply_googly_frame, apply_life_lights, build_googly_track
from spider.viewing.stalk import apply_stalk_presentation, stalk_camera


WIDTH, HEIGHT, FPS = 1080, 1350, 50
PANE_WIDTH, SHOT_SECONDS = WIDTH // 2, 5.0
CAMERAS = ((225.0, -25.0, 3.60), (270.0, -18.0, 3.45))
STABLE = ROOT / "artifacts" / "walk_stable_100" / "replay"
FAST = ROOT / "telemetry" / "tuning" / "20260915-policy-cadence-action-round-01" / "comparison-n00200" / "n00200" / "mean-201"


@dataclass
class Replay:
    directory: Path
    name: str
    speed: float
    model: mujoco.MjModel
    data: mujoco.MjData
    states: np.ndarray
    googly: np.ndarray | None
    origin_x: float


def load_replay(directory: Path, name: str, speed: float, expected: np.ndarray) -> Replay:
    model = mujoco.MjModel.from_binary_path(str(directory / "model.mjb"))
    apply_stalk_presentation(model)
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    if ground >= 0:
        model.geom_matid[ground] = -1
        model.geom_rgba[ground] = (0.115, 0.105, 0.155, 1.0)
    model.vis.rgba.fog[:] = (0.055, 0.050, 0.085, 1.0)
    model.vis.rgba.haze[:] = (0.075, 0.068, 0.105, 1.0)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, PANE_WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)
    data = mujoco.MjData(model)
    with np.load(directory / "states.npz", allow_pickle=False) as archive:
        all_states = archive["states"].copy()
        times = archive["times"].copy()
    indices = _matching_indices(times, expected)
    track = build_googly_track(model, all_states, times)
    states = all_states[indices]
    mujoco.mj_setState(model, data, states[0], STATE)
    return Replay(
        directory, name, speed, model, data, states,
        None if track is None else track[indices], float(data.qpos[0]),
    )


def camera_for(replay: Replay, shot: int) -> mujoco.MjvCamera:
    camera = stalk_camera(replay.model, replay.data)
    camera.azimuth, camera.elevation, camera.distance = CAMERAS[shot]
    camera.lookat[:] = (
        float(replay.data.qpos[0]), float(replay.data.qpos[1]), 0.28,
    )
    return camera


def overlay(canvas: Image.Image, shot: int, time_s: float, displacements: list[float]) -> None:
    fade_layer = Image.new("RGBA", canvas.size)
    fade = ImageDraw.Draw(fade_layer)
    for y in range(300):
        fade.line((0, y, WIDTH, y), fill=(18, 18, 28, int(220 * (1 - y / 300))))
    for y in range(1130, HEIGHT):
        fade.line((0, y, WIDTH, y), fill=(18, 18, 28, int(220 * (y - 1130) / 220)))
    canvas.alpha_composite(fade_layer)
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((54, 45), "walk_stable_100 vs walk_fast_200", font=font(27), fill="#b7b1c3")
    draw.text((50, 94), "speed comparison", font=font(64, bold=True), fill="#eee6db")
    draw.text((54, 185), "measured mean replays · same scale", font=font(34), fill="#eee6db")
    draw.line((54, 253, 134, 253), fill="#f59470", width=3)

    labels = (("stable · 0.318 m/s", "#b7b1c3"), ("n=200 · 0.598 m/s", "#eee6db"))
    for pane, ((label, color), displacement) in enumerate(zip(labels, displacements, strict=True)):
        left = pane * PANE_WIDTH
        draw.rectangle((left, 278, left + PANE_WIDTH, 351), fill=(18, 18, 28, 176))
        draw.text((left + 28, 291), label, font=font(27, bold=pane == 1), fill=color)
        draw.rectangle((left, 1070, left + PANE_WIDTH, 1132), fill=(18, 18, 28, 154))
        draw.text(
            (left + 28, 1082), f"Δx {displacement:.2f} m", font=font(27), fill=color,
        )
    draw.line((PANE_WIDTH, 278, PANE_WIDTH, 1132), fill=(183, 177, 195, 105), width=1)

    view = "three-quarter comparison" if shot == 0 else "side comparison · same replays"
    draw.text((54, 1190), view, font=font(34), fill="#eee6db")
    draw.text((54, 1254), "measured replays · 1× speed · n=200 is 1.88×", font=font(27), fill="#b7b1c3")
    draw.text((1026, 1254), f"{time_s:.2f} s", anchor="ra", font=font(27), fill="#b7b1c3")


def render(destination: Path) -> Path:
    expected = np.arange(round(SHOT_SECONDS * FPS), dtype=np.float64) / FPS
    replays = [
        load_replay(STABLE, "stable", 0.31750619506970296, expected),
        load_replay(FAST, "n=200", 0.5976164914130955, expected),
    ]
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
            renderers = [
                stack.enter_context(mujoco.Renderer(r.model, height=HEIGHT, width=PANE_WIDTH))
                for r in replays
            ]
            for shot in range(2):
                for index, time_s in enumerate(expected):
                    canvas = Image.new("RGBA", (WIDTH, HEIGHT))
                    displacements = []
                    for pane, (replay, renderer) in enumerate(zip(replays, renderers, strict=True)):
                        mujoco.mj_setState(replay.model, replay.data, replay.states[index], STATE)
                        apply_googly_frame(replay.model, replay.googly, index)
                        apply_life_lights(replay.model, float(time_s))
                        mujoco.mj_forward(replay.model, replay.data)
                        renderer.update_scene(replay.data, camera=camera_for(replay, shot))
                        add_reference_grid(renderer)
                        canvas.paste(Image.fromarray(renderer.render()), (pane * PANE_WIDTH, 0))
                        displacements.append(float(replay.data.qpos[0]) - replay.origin_x)
                    overlay(canvas, shot, float(time_s), displacements)
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
    return destination.resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("C:/Users/haidm/Desktop/demos/spider-walk-stable-vs-fast-200.mp4"),
    )
    print(render(parser.parse_args().output))


if __name__ == "__main__":
    main()
