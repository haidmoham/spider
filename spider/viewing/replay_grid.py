"""Four recorded treatments in one synchronized window; no physics integration."""

from pathlib import Path
import textwrap
import time

import mujoco
import numpy as np

from ..recording import STATE


def pane_heading(label):
    """Identify the saved treatment, independent of its quadrant position."""
    name = label.lower()
    if name.startswith('after '):
        return 'YOUR CODE: AFTER TRAINING', 'Saved trained policy', (.38, .18, .04)
    if name.startswith('before '):
        return 'BEFORE TRAINING', 'Same network, initial weights', (.10, .16, .24)
    if name.startswith('neutral control'):
        return 'NEUTRAL CONTROL', 'Neutral targets; no learned actions', (.12, .15, .18)
    if name.startswith('fixed shuffle'):
        return 'FIXED SHUFFLE', 'Hand-coded reference; not the learned policy', (.12, .15, .18)
    return 'RECORDED TREATMENT', 'Saved replay', (.12, .15, .18)


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


def replay_grid(directories, speed=0.5, *, ready=None, screenshot=None, max_frames=None):
    import glfw

    if len(directories) != 4 or not np.isfinite(speed) or speed <= 0:
        raise ValueError('Supply exactly four recordings and a finite positive speed')
    panes = []
    for directory in map(Path, directories):
        model = mujoco.MjModel.from_binary_path(str(directory / 'model.mjb'))
        with np.load(directory / 'states.npz', allow_pickle=False) as archive:
            states, times = archive['states'], archive['times']
            label = str(archive['label'])
        if (states.shape != (len(times), mujoco.mj_stateSize(model, STATE)) or
                len(times) == 0 or not np.isfinite(states).all() or
                not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
            raise ValueError(f'Invalid recorded states/times: {directory}')
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, states[0], STATE)
        mujoco.mj_forward(model, data)
        panes.append(dict(model=model, data=data, states=states, times=times,
                          label=label, origin=data.qpos[:3].copy()))
    if not glfw.init():
        raise RuntimeError('GLFW could not initialize a display')
    window = None
    contexts = []
    try:
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        window = glfw.create_window(1440, 960, 'C-1N | four recorded treatments', None, None)
        if not window:
            raise RuntimeError('Could not create the replay window')
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = (0, 0, .25)
        camera.distance, camera.azimuth, camera.elevation = 1.55, 225, -25
        options = mujoco.MjvOption()
        for pane in panes:
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
                mujoco.mj_forward(model, data)
                mujoco.mjv_updateScene(model, data, options, None, camera,
                                      mujoco.mjtCatBit.mjCAT_ALL, pane['scene'])
                viewport = mujoco.MjrRect(*rectangle)
                mujoco.mjr_render(viewport, pane['scene'], pane['context'])
                title, description, color = pane_heading(pane['label'])
                left, bottom, pane_width, pane_height = rectangle
                header = mujoco.MjrRect(left, bottom + pane_height - 82, pane_width, 82)
                mujoco.mjr_rectangle(header, *color, 1.)
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_BIG,
                                   mujoco.mjtGridPos.mjGRID_TOPLEFT, header,
                                   title, '', pane['context'])
                detail = '\n'.join(textwrap.wrap(pane['label'], width=max(25, pane_width // 9)))
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
