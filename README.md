# Joy Division ParaView Equalizer

A 32-band, audio-reactive 3-D equalizer for **Joy Division — Disorder (1979)**. ParaView loads the precomputed spectrum from `audio/joy/spectrum_output/spectrum.npz`; the programmable source interpolates it at the requested animation timestamp.

## Project layout

```text
audio/joy/
  soundtrack.wav                 source soundtrack
  analyze_spectrum.py            STFT, smoothing, and peak-hold generator
  spectrum_output/spectrum.npz   runtime spectrum cache
state/equalizer.pvsm             portable ParaView scene
animation/render_equalizer.py    deterministic PNG + FFmpeg renderer
output/joydivision_full_hd.mp4   rendered 1080p result
```

## Reproduce

```bash
conda env create -f environment.yml
conda activate pv-beat
export EQUALIZER_ROOT="$PWD"
python audio/joy/analyze_spectrum.py audio/joy/soundtrack.wav \
  --output audio/joy/spectrum_output
pvpython animation/render_equalizer.py
```

The renderer creates exact timestamps `t = frame / 60`, writes 12,543 PNG frames at 1920×1080, and muxes them with AAC audio into `output/joydivision_full_hd.mp4`. The song duration (209.04 s) is not an exact 60-FPS multiple, so the final 10 ms is silent padding; musical content remains synchronized.

## ParaView editing

```bash
export EQUALIZER_ROOT="$PWD"
paraview state/equalizer.pvsm
```

Camera, layout, colors, lighting, title, and labels remain editable in the GUI. Save your variations to a new state file. The state relies on `spectrum_output/spectrum.npz`; regenerate it with the command above after changing the soundtrack or analysis settings.

## Smaller test

```bash
pvpython animation/render_equalizer.py \
  --frames 1000 --width 1280 --height 720 --crf 18 --preset medium \
  --frames-dir renders/test_frames --output output/test.mp4
```
