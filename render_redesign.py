"""Render matched before-and-after views of the C-1N visual layer."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw, ImageFont

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


def contact_sheet(before: dict[str, Image.Image], after: dict[str, Image.Image],
                  before_label: str = "PHYSICS BASELINE") -> Image.Image:
    width, height = 640, 480
    header, row_height = 130, height + 52
    sheet = Image.new("RGB", (width * 2, header + row_height * len(VIEWS)), "#171722")
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.load_default(size=36)
    label_font = ImageFont.load_default(size=18)
    detail_font = ImageFont.load_default(size=16)
    draw.text((24, 22), "C-1N / PORCELAIN SURVEYOR", font=title_font, fill="#ece2d6")
    draw.text((24, 75), "Visual study     /     Matched cameras     /     Canonical reset pose",
              font=detail_font, fill="#aaa6bb")
    draw.line((24, 111, width * 2 - 24, 111), fill="#f59470", width=2)
    for row, name in enumerate(VIEWS):
        top = header + row * row_height
        sheet.paste(before[name], (0, top + 52))
        sheet.paste(after[name], (width, top + 52))
        view_label = name.replace("_", " ").upper()
        draw.text((24, top + 17), f"{before_label} / {view_label}", font=label_font, fill="#aaa6bb")
        draw.text((width + 24, top + 17), f"PORCELAIN / {view_label}", font=label_font, fill="#f59470")
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts") / "c1n_redesign")
    parser.add_argument("--before-directory", type=Path,
                        help="Compare with a prior render's after_*.png images instead of the physics baseline")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    before = ({name: Image.open(args.before_directory / f"after_{name}.png").convert("RGB")
               for name in VIEWS} if args.before_directory else render_views(BASELINE_PATH))
    after = render_views(MODEL_PATH)
    for name in VIEWS:
        before[name].save(args.output / f"before_{name}.png")
        after[name].save(args.output / f"after_{name}.png")
    contact_sheet(before, after, "PREVIOUS DESIGN" if args.before_directory else "PHYSICS BASELINE").save(args.output / "comparison.png")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
