"""Render matched before-and-after views of the C-1N visual layer."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw

from simulation import MODEL_PATH, reset
from test_visual_invariants import BASELINE_PATH


VIEWS = {
    "front": (180.0, -15.0),
    "side": (270.0, -15.0),
    "top": (90.0, -90.0),
    "three_quarter": (225.0, -25.0),
}


def render_views(model_path: Path) -> dict[str, Image.Image]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    reset(model, data)
    images: dict[str, Image.Image] = {}
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for name, (azimuth, elevation) in VIEWS.items():
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.lookat[:] = (0.0, 0.0, 0.25)
            camera.distance = 1.55
            camera.azimuth = azimuth
            camera.elevation = elevation
            renderer.update_scene(data, camera=camera)
            images[name] = Image.fromarray(renderer.render())
    return images


def contact_sheet(before: dict[str, Image.Image], after: dict[str, Image.Image]) -> Image.Image:
    width, height = 640, 480
    sheet = Image.new("RGB", (width * 2, (height + 42) * len(VIEWS)), "#101414")
    draw = ImageDraw.Draw(sheet)
    for row, name in enumerate(VIEWS):
        top = row * (height + 42)
        sheet.paste(before[name], (0, top + 42))
        sheet.paste(after[name], (width, top + 42))
        draw.text((18, top + 13), f"BEFORE / {name.upper()}", fill="#ddd8c5")
        draw.text((width + 18, top + 13), f"GOOGLY SURVEYOR / {name.upper()}", fill="#f04b25")
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts") / "c1n_redesign")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    before = render_views(BASELINE_PATH)
    after = render_views(MODEL_PATH)
    for name in VIEWS:
        before[name].save(args.output / f"before_{name}.png")
        after[name].save(args.output / f"after_{name}.png")
    contact_sheet(before, after).save(args.output / "comparison.png")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
