#!/usr/bin/env pvpython
"""Render the ParaView equalizer at exact timestamps and mux its soundtrack."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, default=ROOT / "state" / "equalizer.pvsm")
    parser.add_argument("--audio", type=Path, default=ROOT / "audio" / "joy" / "soundtrack.wav")
    parser.add_argument("--source", help="optional single source/filter name; default updates the whole saved pipeline")
    parser.add_argument("--frames", type=int, default=12543)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, help="optional output width; omit to preserve saved view size")
    parser.add_argument("--height", type=int, help="optional output height; omit to preserve saved view size")
    parser.add_argument("--frames-dir", type=Path, default=ROOT / "renders" / "frames")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "joydivision_full_hd.mp4")
    parser.add_argument("--crf", type=int, default=14)
    parser.add_argument("--preset", default="slow")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.frames < 1 or args.fps < 1:
        raise ValueError("frames and fps must be positive")
    if not args.state.is_file() or not args.audio.is_file():
        raise FileNotFoundError("State or audio file is missing")
    os.environ["EQUALIZER_ROOT"] = str(ROOT)
    duration = args.frames / args.fps
    frames_dir = args.frames_dir.resolve()
    output = args.output.resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = list(frames_dir.glob("frame_*.png"))
    if existing and not args.overwrite:
        raise FileExistsError(f"{frames_dir} already has frames; use --overwrite to replace them")

    from paraview.simple import (_DisableFirstRenderCameraReset, FindSource, GetAnimationScene,
                                 GetRenderViews, GetSources, LoadState, Render,
                                 SaveScreenshot, UpdatePipeline)
    _DisableFirstRenderCameraReset()
    LoadState(str(args.state.resolve()))  # Input PVSM is never saved or changed.
    render_views = GetRenderViews()
    if len(render_views) != 1:
        raise RuntimeError(f"Expected exactly one saved Render View; found {len(render_views)}")
    view = render_views[0]
    if (args.width is None) != (args.height is None):
        raise ValueError("Specify both --width and --height, or neither")
    if args.width is not None:
        view.ViewSize = [args.width, args.height]

    def camera_parameters():
        return {"position": list(view.CameraPosition), "focal_point": list(view.CameraFocalPoint),
                "view_up": list(view.CameraViewUp), "parallel_scale": float(view.CameraParallelScale),
                "view_angle": float(view.CameraViewAngle), "view_size": list(view.ViewSize)}

    print("Saved camera after LoadState:", camera_parameters())
    scene = GetAnimationScene()
    if args.source:
        source = FindSource(args.source)
        if source is None:
            raise RuntimeError(f"No source named {args.source!r} in {args.state}")
        update_targets = [source]
    else:
        # A visible Threshold can depend on a hidden Programmable Source.  Update
        # every object so the rendered scene matches the complete GUI pipeline.
        update_targets = list(GetSources().values())
        if not update_targets:
            raise RuntimeError("The saved state has no pipeline objects")
    print("Updating pipeline objects:", [proxy.GetXMLLabel() for proxy in update_targets])
    for frame in range(args.frames):
        timestamp = frame / args.fps
        # Apply Time Manager cues (opacity, camera, etc.) before data updates.
        scene.AnimationTime = timestamp
        for source in update_targets:
            UpdatePipeline(time=timestamp, proxy=source)
        view.ViewTime = timestamp
        if frame == 0:
            print("Camera after Time Manager cue at first frame:", camera_parameters())
        Render(view)
        screenshot_args = ({"ImageResolution": [args.width, args.height]}
                           if args.width is not None else {})
        SaveScreenshot(str(frames_dir / f"frame_{frame:06d}.png"), view, **screenshot_args)
        if frame == 0:
            print("Saved camera after first SaveScreenshot:", camera_parameters())
        if (frame + 1) % 100 == 0 or frame == 0:
            print(f"Rendered {frame + 1}/{args.frames} at {timestamp:.6f} s")

    command = ["ffmpeg", "-y", "-framerate", str(args.fps), "-start_number", "0",
               "-i", str(frames_dir / "frame_%06d.png"), "-ss", "0", "-t", f"{duration:.9f}",
               "-i", str(args.audio.resolve()), "-frames:v", str(args.frames),
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-crf", str(args.crf), "-preset", args.preset,
               "-pix_fmt", "yuv420p", "-r", str(args.fps), "-c:a", "aac", "-b:a", "192k",
               "-af", f"apad,atrim=duration={duration:.9f}", "-movflags", "+faststart", str(output)]
    subprocess.run(command, check=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                            "-show_entries", "stream=nb_read_frames,avg_frame_rate", "-of", "json", str(output)],
                           check=True, capture_output=True, text=True)
    result = json.loads(probe.stdout)["streams"][0]
    if int(result["nb_read_frames"]) != args.frames or result["avg_frame_rate"] != f"{args.fps}/1":
        raise RuntimeError("Encoded MP4 did not preserve requested frame count/FPS")
    print(f"Wrote {output} ({args.frames} frames at {args.fps} FPS; {duration:.6f} s)")


if __name__ == "__main__":
    main()
