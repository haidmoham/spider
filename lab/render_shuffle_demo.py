"""Render a styled five-second video from recorded C-1N states, without physics steps."""
from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--recording', type=Path, required=True, help='Directory containing model.mjb and states.npz')
parser.add_argument('--output', type=Path, default=ROOT / 'telemetry/exports')
args = parser.parse_args()
SOURCE = args.recording.resolve()
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)
SIZE, FPS, FRAMES, SPEED = 1080, 30, 150, 0.8
VIDEO = OUT / 'c1n-shuffle-social.mp4'
archive = np.load(SOURCE / 'states.npz', allow_pickle=False)
states, times = archive['states'], archive['times']
if len(times) < 2 or not np.isfinite(states).all() or not np.all(np.diff(times) > 0):
    raise ValueError('Replay must contain finite states and increasing timestamps.')
if times[-1] - times[0] < 4.0 - 1e-9:
    raise ValueError('Five-second export at 0.8x requires at least four seconds of recorded states.')
model = mujoco.MjModel.from_binary_path(str(SOURCE / 'model.mjb'))
model.vis.global_.offwidth = model.vis.global_.offheight = SIZE
data = mujoco.MjData(model)
camera = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(camera)
camera.lookat[:] = (0.06, 0.0, 0.25)
camera.distance, camera.azimuth, camera.elevation = 1.85, 225, -25
state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
mujoco.mj_setState(model, data, states[0], state_spec)
origin_x = float(data.qpos[0])

FONTS = Path('C:/Windows/Fonts')
headline = ImageFont.truetype(str(FONTS / 'arialbd.ttf'), 112)
serif = ImageFont.truetype(str(FONTS / 'georgiai.ttf'), 38)
label_font = ImageFont.truetype(str(FONTS / 'consola.ttf'), 19)
number_font = ImageFont.truetype(str(FONTS / 'bahnschrift.ttf'), 57)
small = ImageFont.truetype(str(FONTS / 'consola.ttf'), 17)
PAPER, MUTED, ACCENT, LINE = '#ece6dc', '#a7a4b2', '#ed9b79', '#454353'

def add_grid(scene):
    # Display geometry only; no masses, contacts or dynamics are changed.
    for axis in (0, 1):
        for value in np.arange(-1.2, 1.21, 0.1):
            start, end = np.array([-1.2, value, 0.001]), np.array([1.2, value, 0.001])
            if axis:
                start[[0, 1]], end[[0, 1]] = start[[1, 0]], end[[1, 0]]
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3),
                               np.zeros(3), np.eye(3).ravel(), np.array([0.24, 0.225, 0.28, 0.5]))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 1.0, start, end)
            scene.ngeom += 1

def compose(rgb, frame, recorded_time, displacement):
    im = Image.fromarray(rgb).convert('RGBA')
    overlay = Image.new('RGBA', im.size)
    draw = ImageDraw.Draw(overlay)
    # Soft top/bottom shade connects the type to the stage without a panel border.
    for y in range(300):
        draw.line((0, y, SIZE, y), fill=(17, 17, 28, int(155 * (1-y/300)**1.4)))
    for y in range(790, SIZE):
        draw.line((0, y, SIZE, y), fill=(17, 17, 28, int(225 * ((y-790)/290)**0.6)))
    im = Image.alpha_composite(im, overlay)
    draw = ImageDraw.Draw(im)
    draw.text((56, 38), 'ROBOTICS / MOTION STUDY', font=label_font, fill=MUTED)
    draw.text((1024, 38), 'MUJOCO', font=label_font, fill=MUTED, anchor='ra')
    draw.text((49, 65), 'C–1N', font=headline, fill=PAPER)
    draw.text((58, 207), 'a small shuffle.', font=serif, fill=PAPER)
    draw.line((58, 274, 128, 274), fill=ACCENT, width=3)
    draw.text((1024, 217), '6 LEGS\n18 JOINTS', font=label_font, fill=MUTED,
              anchor='ra', align='right', spacing=8)

    draw.line((56, 879, 1024, 879), fill=LINE, width=1)
    draw.text((56, 900), 'FORWARD DISPLACEMENT', font=small, fill=MUTED)
    draw.text((54, 927), f'{displacement * 100:04.1f}', font=number_font, fill=PAPER)
    draw.text((210, 954), 'cm', font=label_font, fill=ACCENT)
    draw.text((415, 900), 'SIMULATION TIME', font=small, fill=MUTED)
    draw.text((413, 927), f'{recorded_time:04.2f}', font=number_font, fill=PAPER)
    draw.text((557, 954), 's', font=label_font, fill=ACCENT)
    draw.text((1024, 900), 'RECORDED PHYSICS', font=small, fill=MUTED, anchor='ra')
    draw.text((1024, 943), '0.8× PLAYBACK', font=label_font, fill=PAPER, anchor='ra')
    draw.line((56, 1023, 1024, 1023), fill=LINE, width=2)
    progress_x = 56 + int(968 * frame / (FRAMES-1))
    draw.line((56, 1023, progress_x, 1023), fill=ACCENT, width=3)
    draw.ellipse((progress_x-4, 1019, progress_x+4, 1027), fill=PAPER)
    return im.convert('RGB')

cmd = [shutil.which('ffmpeg'), '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
       '-s', f'{SIZE}x{SIZE}', '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264',
       '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(VIDEO)]
previews = []
with (OUT / 'ffmpeg-social.log').open('w') as log:
    encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log, stderr=log)
    try:
        with mujoco.Renderer(model, height=SIZE, width=SIZE) as renderer:
            for frame in range(FRAMES):
                desired_time = times[0] + frame / FPS * SPEED
                index = min(len(times)-1, max(0, int(np.searchsorted(times, desired_time, side='right'))-1))
                mujoco.mj_setState(model, data, states[index], state_spec)
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                add_grid(renderer.scene)
                image = compose(renderer.render(), frame, float(times[index]-times[0]), float(data.qpos[0])-origin_x)
                encoder.stdin.write(image.tobytes())
                if frame in (0, 50, 100, 149):
                    image.save(OUT / f'c1n-social-frame-{frame:03}.jpg', quality=94)
                    previews.append(image.resize((540, 540)))
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError('FFmpeg failed')
    except BaseException:
        encoder.kill()
        encoder.wait()
        raise
sheet = Image.new('RGB', (1080, 1080))
for i, preview in enumerate(previews):
    sheet.paste(preview, ((i % 2) * 540, (i // 2) * 540))
sheet.save(OUT / 'c1n-social-contact-sheet.jpg', quality=94)
(OUT / 'c1n-shuffle-social.json').write_text(json.dumps(dict(
    source=str(SOURCE), label=str(archive['label']), playback_speed=SPEED,
    frames=FRAMES, fps=FPS, seconds=5, resolution=[SIZE, SIZE],
    presentation='Locked camera, render-only floor grid, measured displacement and time.',
    physics='Exact saved states; no integration, no changed gait or model dynamics.',
), indent=2), encoding='utf-8')
print(VIDEO)
