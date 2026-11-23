#!/usr/bin/env pvpython
"""Render the saved Unknown Pleasures layout and mux its WAV at 60 FPS."""
from __future__ import annotations

import json
import argparse
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state" / "unknown_pleasures.pvsm"
AUDIO = ROOT / "audio" / "joy" / "soundtrack.wav"
FRAMES_DIR = ROOT / "renders" / "unknown_pleasures_frames"
OUTPUT = ROOT / "output" / "unknown_pleasures_60fps.mp4"
FPS = 60
FRAMES = 12_543  # ceil(209.039771 s * 60 FPS)
RESOLUTION = [1920, 1080]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=FRAMES)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if not (0 <= args.start <= args.end <= FRAMES):
        raise ValueError(f"Frame range must be within 0..{FRAMES}")
    if args.render_only and args.encode_only:
        raise ValueError("Choose only one of --render-only and --encode-only")

    if not args.encode_only:
        render_frames(args.start, args.end, quiet=args.quiet)
    if not args.render_only:
        encode_movie(quiet=args.quiet)


def render_frames(start: int, end: int, quiet: bool = False) -> None:
    from paraview.simple import (
        GetAnimationScene,
        GetLayout,
        GetRenderViews,
        GetSources,
        LoadState,
        RenderAllViews,
        SaveScreenshot,
        UpdatePipeline,
        _DisableFirstRenderCameraReset,
    )

    _DisableFirstRenderCameraReset()
    LoadState(str(STATE))
    layout = GetLayout()
    views = GetRenderViews()
    sources = list(GetSources().values())
    if layout is None or len(views) != 2 or not sources:
        raise RuntimeError("The saved two-view layout or pipeline could not be loaded")

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    scene = GetAnimationScene()
    for frame in range(start, end):
        timestamp = frame / FPS
        # Apply the state-file cues (including the saved camera path), then
        # update every pipeline object at the exact audio timestamp.
        scene.AnimationTime = timestamp
        for source in sources:
            UpdatePipeline(time=timestamp, proxy=source)
        for view in views:
            view.ViewTime = timestamp
        RenderAllViews()
        SaveScreenshot(
            str(FRAMES_DIR / f"frame_{frame:06d}.png"),
            layout,
            ImageResolution=RESOLUTION,
        )
        if not quiet and (frame == start or (frame + 1) % 100 == 0):
            print(f"Rendered frame {frame + 1}/{FRAMES} at {timestamp:.6f} s", flush=True)


def encode_movie(quiet: bool = False) -> None:
    missing = [
        frame
        for frame in range(FRAMES)
        if not (FRAMES_DIR / f"frame_{frame:06d}.png").is_file()
        or (FRAMES_DIR / f"frame_{frame:06d}.png").stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"Cannot encode: {len(missing)} rendered frames are missing")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    duration = FRAMES / FPS
    ffmpeg_output = {
        "stdout": subprocess.DEVNULL if quiet else None,
        "stderr": subprocess.DEVNULL if quiet else None,
    }
    # Some FFmpeg/PNG builds fail after a long continuous image sequence. Encode
    # short, exact-CFR sections and concatenate their elementary streams instead.
    with tempfile.TemporaryDirectory(prefix="unknown_pleasures_", dir=OUTPUT.parent) as temp_dir:
        temp = Path(temp_dir)
        sections = []
        for section, start in enumerate(range(0, FRAMES, 1000)):
            frame_count = min(1000, FRAMES - start)
            part = temp / f"part_{section:03d}.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-framerate", str(FPS), "-start_number", str(start),
                    "-i", str(FRAMES_DIR / "frame_%06d.png"), "-frames:v", str(frame_count),
                    "-an", "-c:v", "libx264", "-crf", "14", "-preset", "slow",
                    "-pix_fmt", "yuv420p", "-r", str(FPS), str(part),
                ],
                check=True,
                **ffmpeg_output,
            )
            sections.append(part)
        manifest = temp / "sections.txt"
        manifest.write_text("".join(f"file '{part}'\n" for part in sections))
        video = temp / "video.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(manifest),
             "-c", "copy", "-movflags", "+faststart", str(video)],
            check=True,
            **ffmpeg_output,
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video), "-ss", "0", "-t", f"{duration:.9f}",
                "-i", str(AUDIO), "-frames:v", str(FRAMES), "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-af", f"apad,atrim=duration={duration:.9f}",
                "-movflags", "+faststart", str(OUTPUT),
            ],
            check=True,
            **ffmpeg_output,
        )
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,avg_frame_rate",
            "-of", "json", str(OUTPUT),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    if int(stream["nb_read_frames"]) != FRAMES or stream["avg_frame_rate"] != "60/1":
        raise RuntimeError("FFmpeg did not preserve the requested 60-FPS frame sequence")
    if not quiet:
        print(f"Wrote {OUTPUT} ({FRAMES} frames at {FPS} FPS; {duration:.6f} s)")


if __name__ == "__main__":
    main()
