# C-1N demo style

Use `spider-notebook-04-vibrating-baseline.mp4` as the visual reference for short
checkpoint films. Match its 1080 × 1350 frame, 50 fps, ten-second duration,
fixed header and footer gradients, warm off-white Segoe UI type, coral rule,
purple stage, thin continuous 0.25 m world grid, and two equal five-second
shots. Use azimuth 225° and elevation −25° for the first shot. Cut to azimuth
270° and elevation −18° at five seconds. Increase only the camera distance that
the current chassis needs to keep all six feet in frame. Reset the visible
replay clock after the cut.

Render saved MuJoCo states at 1×. Apply the repository's stalk presentation,
dynamic life lights, and gravity-responsive googly-eye track. Keep the header
copy specific to the checkpoint. Keep the lower caption factual and identify a
replay as measured when its states came from a recorded rollout. Do not imply
that repeating the same replay from a second camera is a second trial.

The `walk_stable_100` example is reproducible with:

```powershell
.venv\Scripts\python scripts\render_walk_demo.py
```

For speed comparisons, use synchronized split-screen panels. Use the same
camera scale, clock, and 1× playback in both panels. Show measured speed and
live displacement beside each replay. Repeat both recordings from the second
camera after five seconds. See `scripts/render_walk_speed_demo.py`.

Generated videos belong outside Git. Verify the final duration, dimensions,
frame rate, video codec, pixel format, representative frames, transition, and
audio stream. Preserve silence when the reference has no audio.
