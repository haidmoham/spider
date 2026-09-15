"""Four recorded treatments in one synchronized window; no physics integration."""

from pathlib import Path
import re
import textwrap
import time

import mujoco
import numpy as np

from ..recording import STATE


def pane_heading(label, training_updates=None):
    """Identify the saved treatment, independent of its quadrant position."""
    name = label.lower()
    if 'walk_stable_100' in name:
        return ('walk_stable_100 / LOCKED', 'Approved 1.1 Hz walking baseline', (.14, .20, .16))
    if name.startswith('reference ppo n='):
        fields = [part.strip() for part in label.split('|')]
        if fields[1] == 'LEARNED CADENCE':
            return (f'{fields[0].replace("REFERENCE ", "GAIT ")} / {fields[2].upper()}',
                    'Policy chooses cadence and joint corrections', (.20, .10, .12))
        cadence = re.search(r'cadence=([\d.]+) Hz', label)
        if cadence:
            return (f'{fields[0].replace("REFERENCE ", "")} / {cadence.group(1)} Hz / {fields[2].upper()}',
                    'Cadence treatment; measured evaluation', (.20, .10, .12))
        return (f'{fields[0]} / {fields[2].upper()}',
                'Candidate chassis; measured evaluation; not promoted', (.20, .10, .12))
    if 'kinematic' in name and 'untrained' in name:
        gait = label.split('/', 1)[0].strip()
        return (f'{gait} / KINEMATIC / UNTRAINED',
                'Prescribed pose preview; no dynamics claim', (.28, .12, .12))
    if 'open-loop' in name and 'feasibility' in name:
        return ('OPEN-LOOP FEASIBILITY / MEASURED',
                'Recorded dynamics; no learned-policy claim', (.14, .20, .16))
    if name.startswith('recorded accepted ppo-100'):
        return ('ACCEPTED PPO-100 / MEASURED',
                'Recorded dynamics comparator', (.13, .17, .20))
    if name.startswith('ppo n='):
        fields = [part.strip() for part in label.split('|')]
        treatment = fields[1].lower() if len(fields) > 1 else ''
        tuning_labels = {
            'lower_exploration': ('LOWER EXPLORATION / n=50', 'Noise x0.4; no entropy bonus; target 0.25 m/s'),
            'stronger_forward': ('STRONGER FORWARD / n=50', 'Target 0.4 m/s; forward reward weight 4'),
            'combined': ('COMBINED / n=50', 'Lower exploration + stronger forward reward'),
            'frozen_lower': ('FROZEN LOWER / n=100', 'Parent policy; no continuation'),
            'control': ('CONTINUED CONTROL / n=110', '10 updates; no command-rate penalty'),
            'rate_small': ('SMALL PENALTY / n=110', '10 updates; rate weight 3.73116'),
            'rate_medium': ('MEDIUM PENALTY / n=110', '10 updates; rate weight 9.32791'),
        }
        if fields[-1] in tuning_labels:
            title, description = tuning_labels[fields[-1]]
            return title, description, (.13, .17, .20)
        if fields[-1] == 'accepted_baseline':
            return 'ACCEPTED BASELINE / n=100', 'Frozen comparison policy', (.13, .17, .20)
        stride_seed = re.fullmatch(r'stride_seed_(\d+)', fields[-1])
        if stride_seed:
            return (f'FRESH STRIDE / seed={stride_seed.group(1)} / n=50',
                    'Phase-guided training; motion quality under review', (.18, .13, .22))
        descriptions = {
            'baseline': 'Accepted policy; original action mapping',
            'lower': 'Unvalidated: hip/knee posture ramp',
            'smooth': 'Unvalidated: command filter only',
            'stalk': 'Unvalidated: posture ramp + command filter',
        }
        if treatment in descriptions:
            return f'{fields[0]} / {treatment.upper()}', descriptions[treatment], (.13, .17, .20)
        return fields[0], 'Saved PPO checkpoint', (.13, .17, .20)
    if name.startswith('after '):
        title = 'YOUR CODE: AFTER TRAINING'
        if training_updates is not None:
            title = f'YOUR CODE: n={training_updates} training updates'
        return title, 'Saved trained policy', (.38, .18, .04)
    if name.startswith('before '):
        return 'BEFORE TRAINING (n=0)', 'Same network, initial weights', (.10, .16, .24)
    if name.startswith('neutral control'):
        return 'NEUTRAL CONTROL', 'Neutral targets; no learned actions', (.12, .15, .18)
    if name.startswith('fixed shuffle'):
        return 'FIXED SHUFFLE', 'Hand-coded reference; not the learned policy', (.12, .15, .18)
    return 'RECORDED TREATMENT', 'Saved replay', (.12, .15, .18)


def recorded_training_updates(directory):
    """Read the notebook's saved evaluation count; unknown paths stay unknown."""
    for parent in (Path(directory), *Path(directory).parents):
        match = re.fullmatch(r'evaluation-(\d+)-\d+', parent.name)
        if match:
            return int(match.group(1))
    return None


def quadrants(width, height, gap=4):
    """Top-left, top-right, bottom-left, bottom-right in framebuffer coordinates."""
    half_w, half_h = width // 2, height // 2
    return [(0, half_h + gap, half_w - gap, height - half_h - gap),
            (half_w + gap, half_h + gap, width - half_w - gap, height - half_h - gap),
            (0, 0, half_w - gap, half_h - gap),
            (half_w + gap, 0, width - half_w - gap, half_h - gap)]


def frame_index(times, elapsed):
    """Use relative simulation time; hold the final frame of shorter episodes."""
    return max(0, min(len(times) - 1,
                      int(np.searchsorted(times, times[0] + elapsed, side='right')) - 1))


def replay_grid(directories, speed=0.5, *, ready=None, screenshot=None, max_frames=None,
                presentation='original'):
    import glfw

    if len(directories) != 4 or not np.isfinite(speed) or speed <= 0:
        raise ValueError('Supply exactly four recordings and a finite positive speed')
    if presentation not in ('original', 'stalk'):
        raise ValueError('Unknown presentation')
    panes = []
    for directory in map(Path, directories):
        model = mujoco.MjModel.from_binary_path(str(directory / 'model.mjb'))
        with np.load(directory / 'states.npz', allow_pickle=False) as archive:
            states, times = archive['states'], archive['times']
            label = str(archive['label'])
        if 'walk_stable_100' in directory.parts:
            label = 'REFERENCE PPO n=100 | walk_stable_100 | mean | seed=201'
        if (states.shape != (len(times), mujoco.mj_stateSize(model, STATE)) or
                len(times) == 0 or not np.isfinite(states).all() or
                not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
            raise ValueError(f'Invalid recorded states/times: {directory}')
        data = mujoco.MjData(model)
        from .googly import build_googly_track
        googly_track = build_googly_track(model, states, times)
        mujoco.mj_setState(model, data, states[0], STATE)
        mujoco.mj_forward(model, data)
        panes.append(dict(model=model, data=data, states=states, times=times,
                          label=label, training_updates=recorded_training_updates(directory),
                          origin=data.qpos[:3].copy(), googly_track=googly_track))
    if not glfw.init():
        raise RuntimeError('GLFW could not initialize a display')
    window = None
    contexts = []
    try:
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        checkpoints = sorted({p['label'].split('|')[0].strip() for p in panes
                              if p['label'].lower().startswith('reference ppo n=')})
        window_title = 'C-1N | ' + (' + '.join(checkpoints) if checkpoints else 'four recorded treatments')
        window = glfw.create_window(1440, 960, window_title, None, None)
        if not window:
            raise RuntimeError('Could not create the replay window')
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = (0, 0, .25)
        camera.distance, camera.azimuth, camera.elevation = 1.55, 225, -25
        if presentation == 'stalk':
            from .stalk import apply_stalk_presentation, stalk_camera
            camera = stalk_camera(panes[0]['model'], panes[0]['data'])
        options = mujoco.MjvOption()
        for pane in panes:
            if presentation == 'stalk':
                apply_stalk_presentation(pane['model'])
            pane['scene'] = mujoco.MjvScene(pane['model'], maxgeom=10000)
            pane['context'] = mujoco.MjrContext(pane['model'], mujoco.mjtFontScale.mjFONTSCALE_100)
            contexts.append(pane['context'])
        playback = dict(elapsed=0., paused=False)

        def key_callback(window, key, scancode, action, mods):
            if action not in (glfw.PRESS, glfw.REPEAT):
                return
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_SPACE and action == glfw.PRESS:
                playback['paused'] = not playback['paused']
            elif key == glfw.KEY_R:
                playback['elapsed'] = 0.
            elif key == glfw.KEY_LEFT:
                camera.azimuth -= 5
            elif key == glfw.KEY_RIGHT:
                camera.azimuth += 5
            elif key == glfw.KEY_UP:
                camera.elevation = min(0, camera.elevation + 5)
            elif key == glfw.KEY_DOWN:
                camera.elevation = max(-90, camera.elevation - 5)
            elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
                camera.distance = max(.3, camera.distance * .9)
            elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
                camera.distance = min(10., camera.distance / .9)

        glfw.set_key_callback(window, key_callback)
        duration = max(float(p['times'][-1] - p['times'][0]) for p in panes)
        previous = time.monotonic()
        frames = 0
        while not glfw.window_should_close(window):
            now = time.monotonic()
            if not playback['paused']:
                playback['elapsed'] = (playback['elapsed'] + (now - previous) * speed) % (duration + speed)
            previous = now
            width, height = glfw.get_framebuffer_size(window)
            if width < 20 or height < 20:
                glfw.wait_events_timeout(.05)
                continue
            mujoco.mjr_rectangle(mujoco.MjrRect(0, 0, width, height), .08, .08, .1, 1.)
            for pane, rectangle in zip(panes, quadrants(width, height)):
                index = frame_index(pane['times'], playback['elapsed'])
                model, data = pane['model'], pane['data']
                mujoco.mj_setState(model, data, pane['states'][index], STATE)
                from .googly import apply_googly_frame, apply_life_lights
                apply_googly_frame(model, pane['googly_track'], index)
                apply_life_lights(model, data.time)
                mujoco.mj_forward(model, data)
                if presentation == 'stalk':
                    # Camera-only follow preserves the recorded trajectory.
                    camera.lookat[:] = (float(data.qpos[0]) + .10, float(data.qpos[1]), .24)
                mujoco.mjv_updateScene(model, data, options, None, camera,
                                      mujoco.mjtCatBit.mjCAT_ALL, pane['scene'])
                viewport = mujoco.MjrRect(*rectangle)
                mujoco.mjr_render(viewport, pane['scene'], pane['context'])
                title, description, color = pane_heading(pane['label'], pane['training_updates'])
                left, bottom, pane_width, pane_height = rectangle
                header = mujoco.MjrRect(left, bottom + pane_height - 82, pane_width, 82)
                mujoco.mjr_rectangle(header, *color, 1.)
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_BIG,
                                   mujoco.mjtGridPos.mjGRID_TOPLEFT, header,
                                   title, '', pane['context'])
                detail_label = pane['label']
                if detail_label.lower().startswith('reference ppo n='):
                    detail_label = ' | '.join(detail_label.split('|')[1:]).strip()
                detail = '\n'.join(textwrap.wrap(detail_label, width=max(25, pane_width // 9)))
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,
                                   mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, header,
                                   description + '\n' + detail, '', pane['context'])
                ended = playback['elapsed'] >= pane['times'][-1] - pane['times'][0]
                dx, dy, _ = data.qpos[:3] - pane['origin']
                text = (f"t={pane['times'][index]:.2f}s  dx={dx:+.3f}m  dy={dy:+.3f}m\n"
                        f"{speed:g}x | {'PAUSED' if playback['paused'] else 'END HOLD' if ended else 'PLAY'}\n"
                        'Space pause | R restart | arrows orbit | +/- zoom | Esc close')
                mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_100,
                                   mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, viewport, text, '', pane['context'])
            if screenshot is not None and frames == 0:
                from PIL import Image
                pixels = np.empty((height, width, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(pixels, None, mujoco.MjrRect(0, 0, width, height), contexts[0])
                Image.fromarray(pixels[::-1]).save(screenshot)
            glfw.swap_buffers(window)
            glfw.poll_events()
            frames += 1
            if ready is not None and frames == 1:
                Path(ready).touch()
            if max_frames is not None and frames >= max_frames:
                break
    finally:
        for context in contexts:
            context.free()
        if window:
            glfw.destroy_window(window)
        glfw.terminate()
