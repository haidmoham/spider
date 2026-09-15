"""Render the locked walk_stable_100 replay in the repository demo style."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
import shutil
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spider.recording import STATE
from spider.viewing.film import _matching_indices
from spider.viewing.googly import apply_googly_frame, apply_life_lights, build_googly_track
from spider.viewing.stalk import apply_stalk_presentation, stalk_camera


REPLAY = ROOT / "artifacts" / "walk_stable_100" / "replay"
WIDTH, HEIGHT, FPS = 1080, 1350, 50
SHOT_SECONDS = 5.0
SHOTS = ((225.0, -25.0, 2.02), (270.0, -18.0, 2.18))


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size=size)


def side_camera(model: mujoco.MjModel, data: mujoco.MjData) -> mujoco.MjvCamera:
    camera = stalk_camera(model, data)
    camera.azimuth, camera.elevation, camera.distance = SHOTS[1]
    camera.lookat[:] = (float(data.qpos[0]), float(data.qpos[1]), 0.28)
    return camera


def three_quarter_camera(model: mujoco.MjModel, data: mujoco.MjData) -> mujoco.MjvCamera:
    camera = stalk_camera(model, data)
    camera.azimuth, camera.elevation, camera.distance = SHOTS[0]
    camera.lookat[:] = (float(data.qpos[0]), float(data.qpos[1]), 0.28)
    return camera


def draw_overlay(canvas: Image.Image, shot: int, time_s: float) -> None:
    overlay = Image.new("RGBA", canvas.size)
    fade = ImageDraw.Draw(overlay)
    for y in range(300):
        fade.line((0, y, WIDTH, y), fill=(18, 18, 28, int(220 * (1 - y / 300))))
    for y in range(1130, HEIGHT):
        fade.line((0, y, WIDTH, y), fill=(18, 18, 28, int(220 * (y - 1130) / 220)))
    canvas.alpha_composite(overlay)
    draw = ImageDraw.Draw(canvas)
    draw.text((54, 45), "walk_stable_100 / C-1N", font=font(27), fill="#b7b1c3")
    draw.text((50, 94), "stable walking", font=font(64, bold=True), fill="#eee6db")
    draw.text((54, 185), "PPO · 100 updates · 1.1 Hz", font=font(34), fill="#eee6db")
    draw.line((54, 253, 134, 253), fill="#f59470", width=3)

    label = "three-quarter view" if shot == 0 else "side view · same replay"
    draw.text((54, 1190), label, font=font(34), fill="#eee6db")
    draw.text(
        (54, 1254),
        "measured replay · 1× speed · 0/24 falls",
        font=font(27),
        fill="#b7b1c3",
    )
    clock = f"{time_s:.2f} s"
    draw.text((1026, 1254), clock, anchor="ra", font=font(27), fill="#b7b1c3")


def add_reference_grid(renderer: mujoco.Renderer) -> None:
    """Add the reference film's thin world grid after scene construction."""
    for axis in (0, 1):
        for value in np.arange(-4, 4.01, 0.25):
            start = np.array([-4, value, 0.001])
            end = np.array([4, value, 0.001])
            if axis:
                start = start[[1, 0, 2]]
                end = end[[1, 0, 2]]
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).ravel(),
                np.array([0.29, 0.28, 0.35, 0.55]),
            )
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 1, start, end)
            renderer.scene.ngeom += 1


def render(destination: Path) -> Path:
    model = mujoco.MjModel.from_binary_path(str(REPLAY / "model.mjb"))
    data = mujoco.MjData(model)
    apply_stalk_presentation(model)
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    if ground_id >= 0:
        model.geom_matid[ground_id] = -1
        model.geom_rgba[ground_id] = (0.115, 0.105, 0.155, 1.0)
    model.vis.rgba.fog[:] = (0.055, 0.050, 0.085, 1.0)
    model.vis.rgba.haze[:] = (0.075, 0.068, 0.105, 1.0)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)
    with np.load(REPLAY / "states.npz", allow_pickle=False) as archive:
        states = archive["states"].copy()
        times = archive["times"].copy()
    expected = np.arange(round(SHOT_SECONDS * FPS), dtype=np.float64) / FPS
    indices = _matching_indices(times, expected)
    recorded = states[indices]
    googly = build_googly_track(model, states, times)
    googly = None if googly is None else googly[indices]

    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        with ExitStack() as stack:
            renderer = stack.enter_context(mujoco.Renderer(model, height=HEIGHT, width=WIDTH))
            for shot in range(2):
                for frame_index, state in enumerate(recorded):
                    mujoco.mj_setState(model, data, state, STATE)
                    apply_googly_frame(model, googly, frame_index)
                    apply_life_lights(model, float(expected[frame_index]))
                    mujoco.mj_forward(model, data)
                    camera = three_quarter_camera(model, data) if shot == 0 else side_camera(model, data)
                    renderer.update_scene(data, camera=camera)
                    add_reference_grid(renderer)
                    canvas = Image.fromarray(renderer.render()).convert("RGBA")
                    draw_overlay(canvas, shot, float(expected[frame_index]))
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
        "--output",
        type=Path,
        default=Path("C:/Users/haidm/Desktop/demos/spider-walk-stable-100.mp4"),
    )
    args = parser.parse_args()
    print(render(args.output))


if __name__ == "__main__":
    main()
