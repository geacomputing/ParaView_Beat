#!/usr/bin/env python3
"""
Audio STFT Spectrum Analyzer
============================

Features:
    - 32 logarithmic frequency bands
    - Short-Time Fourier Transform (STFT)
    - Global dB normalization
    - Attack / Release smoothing
    - Peak Hold & Decay
    - WAV input
    - CSV, NPZ and PNG outputs

Install:
    pip install numpy scipy matplotlib

Usage:
    python analyze_spectrum.py soundtrack.wav

    python analyze_spectrum.py soundtrack.wav --bands 32 --fps 200

    python analyze_spectrum.py soundtrack.wav --peak-hold 300 --peak-decay 0.5

Output:
    spectrum_output/
        spectrum.npz
        spectrum.csv
        spectrum.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from scipy import signal
from scipy.io import wavfile


# ============================================================
# DEFAULT PARAMETERS
# ============================================================

DEFAULT_BANDS = 32
DEFAULT_FPS = 200

MIN_FREQ = 20.0
MAX_FREQ = 20000.0






DYNAMIC_RANGE_DB = 60.0
NORMALIZATION_PERCENTILE = 99.9

# Peak Hold
PEAK_DECAY_PER_SECOND = 0.5


FFT_WINDOW_MS = 50.0
ATTACK_MS = 5.0
RELEASE_MS = 100.0
PEAK_HOLD_MS = 200.0


# ============================================================
# AUDIO INPUT
# ============================================================

def read_audio(path):

    sample_rate, samples = wavfile.read(path)

    samples = np.asarray(samples)

    if np.issubdtype(samples.dtype, np.integer):

        info = np.iinfo(samples.dtype)

        if info.min < 0:

            scale = float(
                max(abs(int(info.min)), int(info.max))
            )

            samples = samples.astype(np.float64) / scale

        else:

            midpoint = (
                float(info.max) + float(info.min)
            ) / 2.0

            scale = (
                float(info.max) - float(info.min)
            ) / 2.0

            samples = (
                samples.astype(np.float64) - midpoint
            ) / scale

    else:

        samples = samples.astype(np.float64)

    if samples.ndim == 2:

        samples = samples.mean(axis=1)

    if samples.ndim != 1 or samples.size == 0:

        raise ValueError("Invalid WAV file")

    if not np.all(np.isfinite(samples)):

        raise ValueError(
            "Audio contains NaN or infinite values"
        )

    return int(sample_rate), samples


# ============================================================
# ATTACK / RELEASE
# ============================================================

def smooth_envelope(
    values,
    fps,
    attack_ms=ATTACK_MS,
    release_ms=RELEASE_MS
):

    """
    Smooth the amplitude of each frequency band.

    Attack:
        Controls how quickly bars rise.

    Release:
        Controls how quickly bars fall.
    """

    attack = np.exp(
        -1000.0 / (fps * max(attack_ms, 1e-9))
    )

    release = np.exp(
        -1000.0 / (fps * max(release_ms, 1e-9))
    )

    result = np.zeros_like(values)

    previous = np.zeros(
        values.shape[1],
        dtype=np.float64
    )

    for i in range(len(values)):

        current = values[i]

        coefficient = np.where(
            current > previous,
            attack,
            release
        )

        previous = (
            coefficient * previous
            + (1.0 - coefficient) * current
        )

        result[i] = previous

    return result


# ============================================================
# PEAK HOLD & DECAY
# ============================================================

def peak_hold(
    values,
    fps,
    hold_ms=PEAK_HOLD_MS,
    decay_per_second=PEAK_DECAY_PER_SECOND
):

    """
    Calculate Peak Hold indicators independently
    for each frequency band.

    When a new peak occurs:
        - Update peak position.
        - Reset hold timer.

    During hold:
        - Keep peak position unchanged.

    After hold:
        - Peak falls linearly.

    Peak indicators never fall below their
    corresponding amplitude bars.
    """

    peaks = np.zeros_like(values)

    n_bands = values.shape[1]

    last_peak = np.zeros(
        n_bands,
        dtype=np.float64
    )

    hold_timer = np.zeros(
        n_bands,
        dtype=np.float64
    )

    dt = 1.0 / fps

    hold_duration = hold_ms / 1000.0

    decay_step = decay_per_second * dt

    for i, current in enumerate(values):

        # Detect new peaks
        new_peak = current > last_peak

        # Update peaks
        last_peak[new_peak] = current[new_peak]

        # Reset hold timers
        hold_timer[new_peak] = hold_duration

        # Determine which bands are still holding
        holding = hold_timer > 0.0

        # Decay only bands not holding
        decaying = ~holding

        last_peak[decaying] = np.maximum(
            current[decaying],
            last_peak[decaying] - decay_step
        )

        # Save current peak positions
        peaks[i] = last_peak

        # Advance timers
        hold_timer = np.maximum(
            0.0,
            hold_timer - dt
        )

    return peaks


# ============================================================
# STFT
# ============================================================

def calculate_spectrum(
    samples,
    sample_rate,
    fps,
    n_bands
):

    """
    Compute Short-Time Fourier Transform.

    Frequency bands are logarithmically spaced.
    """

    nperseg = round(
        sample_rate * FFT_WINDOW_MS / 1000.0
    )

    nperseg = max(
        2,
        min(nperseg, len(samples))
    )

    hop = max(
        1,
        round(sample_rate / fps)
    )

    hop = min(hop, nperseg)

    frequencies, times, spectrum = signal.stft(
        samples,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg - hop,
        boundary="zeros",
        padded=True
    )

    magnitude = np.abs(spectrum)

    max_frequency = min(
        MAX_FREQ,
        sample_rate / 2.0
    )

    if max_frequency <= MIN_FREQ:

        raise ValueError(
            "Audio sample rate is too low"
        )

    edges = np.geomspace(
        MIN_FREQ,
        max_frequency,
        n_bands + 1
    )

    bands = np.zeros(
        (len(times), n_bands),
        dtype=np.float64
    )

    for band in range(n_bands):

        low = edges[band]
        high = edges[band + 1]

        if band == n_bands - 1:

            mask = (
                (frequencies >= low)
                & (frequencies <= high)
            )

        else:

            mask = (
                (frequencies >= low)
                & (frequencies < high)
            )

        if np.any(mask):

            bands[:, band] = np.sqrt(
                np.mean(
                    magnitude[mask] ** 2,
                    axis=0
                )
            )

        else:

            # Narrow frequency bands can contain no
            # FFT bins, especially at low frequencies.
            #
            # Use the nearest available FFT bin.
            center = np.sqrt(low * high)

            nearest = np.argmin(
                np.abs(frequencies - center)
            )

            bands[:, band] = magnitude[nearest]

    return times, bands, edges


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_spectrum(bands):

    """
    Convert spectral magnitudes into dB.

    Use a shared global reference across all bands,
    preserving their relative energy.
    """

    reference = np.percentile(
        bands,
        NORMALIZATION_PERCENTILE
    )

    reference = max(
        reference,
        1e-12
    )

    db = 20.0 * np.log10(
        np.maximum(bands, 1e-12) / reference
    )

    normalized = np.clip(
        (db + DYNAMIC_RANGE_DB)
        / DYNAMIC_RANGE_DB,
        0.0,
        1.0
    )

    return normalized


# ============================================================
# EXPORT
# ============================================================

def export_results(
    output,
    times,
    amplitudes,
    peaks,
    edges,
    fps
):

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    n_bands = amplitudes.shape[1]

    # --------------------------------------------------------
    # NPZ
    # --------------------------------------------------------

    np.savez_compressed(
        output / "spectrum.npz",

        time=times.astype(np.float64),

        amplitude=amplitudes.astype(np.float32),

        peaks=peaks.astype(np.float32),

        frequency_edges=edges.astype(np.float64),

        frequency_centers=np.sqrt(
            edges[:-1] * edges[1:]
        ),

        fps=np.float64(fps),

        peak_hold_ms=np.float64(PEAK_HOLD_MS),

        peak_decay_per_second=np.float64(
            PEAK_DECAY_PER_SECOND
        )
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    amplitude_headers = [
        f"band_{i:02d}_amplitude"
        for i in range(n_bands)
    ]

    peak_headers = [
        f"band_{i:02d}_peak"
        for i in range(n_bands)
    ]

    headers = (
        ["time_seconds"]
        + amplitude_headers
        + peak_headers
    )

    data = np.column_stack([
        times,
        amplitudes,
        peaks
    ])

    np.savetxt(
        output / "spectrum.csv",
        data,
        delimiter=",",
        header=",".join(headers),
        comments="",
        fmt="%.6f"
    )

    # --------------------------------------------------------
    # PNG: Spectrum heatmap
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(15, 6)
    )

    image = ax.imshow(
        amplitudes.T,
        aspect="auto",
        origin="lower",
        extent=[
            times[0],
            times[-1],
            0,
            n_bands
        ],
        interpolation="nearest",
        vmin=0,
        vmax=1
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency band")

    ax.set_title(
        f"{n_bands}-Band STFT Spectrum"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Normalized amplitude"
    )

    fig.tight_layout()

    fig.savefig(
        output / "spectrum.png",
        dpi=150
    )

    plt.close(fig)

    # --------------------------------------------------------
    # PNG: Peak Hold diagnostic
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(15, 5)
    )

    # Show first 10 seconds
    duration = min(
        10.0,
        float(times[-1])
    )

    mask = times <= duration

    selected_bands = sorted(set([
        0,
        n_bands // 4,
        n_bands // 2,
        3 * n_bands // 4,
        n_bands - 1
    ]))

    for band in selected_bands:

        ax.plot(
            times[mask],
            amplitudes[mask, band],
            linewidth=1.0,
            label=f"Band {band} amplitude"
        )

        ax.plot(
            times[mask],
            peaks[mask, band],
            linewidth=1.2,
            linestyle="--",
            label=f"Band {band} peak"
        )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Normalized amplitude")

    ax.set_title(
        "Peak Hold & Decay — First 10 Seconds"
    )

    ax.set_ylim(0, 1.05)

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        fontsize=8,
        ncol=2
    )

    fig.tight_layout()

    fig.savefig(
        output / "peak_hold.png",
        dpi=150
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "STFT spectrum analyzer with "
            "Attack/Release and Peak Hold"
        )
    )

    parser.add_argument(
        "wav",
        type=Path,
        help="Input WAV audio file"
    )

    parser.add_argument(
        "--bands",
        type=int,
        default=DEFAULT_BANDS,
        help="Number of frequency bands"
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="Spectrum analysis frames per second"
    )

    parser.add_argument(
        "--peak-hold",
        type=float,
        default=PEAK_HOLD_MS,
        help="Peak Hold duration in milliseconds"
    )

    parser.add_argument(
        "--peak-decay",
        type=float,
        default=PEAK_DECAY_PER_SECOND,
        help="Peak decay in normalized units per second"
    )

    parser.add_argument(
        "--attack",
        type=float,
        default=ATTACK_MS,
        help="Attack time in milliseconds"
    )

    parser.add_argument(
        "--release",
        type=float,
        default=RELEASE_MS,
        help="Release time in milliseconds"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "spectrum_output",
        help="Output directory"
    )

    args = parser.parse_args()

    if (
        args.bands < 1
        or args.fps <= 0
        or args.peak_hold < 0
        or args.peak_decay < 0
        or args.attack < 0
        or args.release < 0
    ):

        parser.error(
            "Invalid parameters: bands/fps must be positive; "
            "timing and decay parameters must be nonnegative"
        )

    # --------------------------------------------------------
    # Read audio
    # --------------------------------------------------------

    sample_rate, samples = read_audio(
        args.wav
    )

    duration = len(samples) / sample_rate

    print()
    print("STFT AUDIO ANALYZER")
    print("=" * 50)

    print(
        f"Input:             {args.wav}"
    )

    print(
        f"Sample rate:       {sample_rate} Hz"
    )

    print(
        f"Duration:          {duration:.2f} s"
    )

    print(
        f"Frequency bands:   {args.bands}"
    )

    print(
        f"Analysis FPS:      {args.fps}"
    )

    print(
        f"Attack:            {args.attack} ms"
    )

    print(
        f"Release:           {args.release} ms"
    )

    print(
        f"Peak Hold:         {args.peak_hold} ms"
    )

    print(
        f"Peak Decay:        {args.peak_decay}/s"
    )

    # --------------------------------------------------------
    # STFT
    # --------------------------------------------------------

    print()
    print("Computing STFT...")

    times, bands, edges = calculate_spectrum(
        samples,
        sample_rate,
        args.fps,
        args.bands
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    print("Normalizing spectrum...")

    bands = normalize_spectrum(
        bands
    )

    # --------------------------------------------------------
    # Attack / Release
    # --------------------------------------------------------

    print("Applying Attack / Release...")

    amplitudes = smooth_envelope(
        bands,
        args.fps,
        attack_ms=args.attack,
        release_ms=args.release
    )

    # --------------------------------------------------------
    # Peak Hold
    # --------------------------------------------------------

    print("Computing Peak Hold...")

    peaks = peak_hold(
        amplitudes,
        args.fps,
        hold_ms=args.peak_hold,
        decay_per_second=args.peak_decay
    )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    print("Exporting results...")

    export_results(
        args.output,
        times,
        amplitudes,
        peaks,
        edges,
        args.fps
    )

    print()
    print("RESULTS")
    print("=" * 50)

    print(
        f"Analysis frames:   {len(times)}"
    )

    print(
        f"Amplitude shape:   {amplitudes.shape}"
    )

    print(
        f"Peak shape:        {peaks.shape}"
    )

    print(
        f"Output directory:  {args.output}"
    )

    print()
    print("Files:")
    print("  spectrum.npz")
    print("  spectrum.csv")
    print("  spectrum.png")
    print("  peak_hold.png")

    print()


if __name__ == "__main__":
    main()
