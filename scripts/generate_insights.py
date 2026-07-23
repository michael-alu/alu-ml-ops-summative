"""Generate dataset insight charts and stats for the UI.

Decoupled from training so the UI visuals are reproducible and deployable.
Reads the ESC-10 subset from data/raw and writes PNGs plus insights.json to assets/.
"""

import json
from pathlib import Path
from typing import Dict, List

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DIR: Path = ROOT / "data" / "raw" / "ESC-50-master"
ASSETS_DIR: Path = ROOT / "assets"

SAMPLE_RATE: int = 22050
DURATION: float = 5.0
N_MELS: int = 128
N_FFT: int = 2048
HOP_LENGTH: int = 512

BLUE: str = "#2a78d6"
INK: str = "#0b0b0b"
INK_SOFT: str = "#52514e"
MUTED: str = "#898781"
GRID: str = "#e1e0d9"
AXIS: str = "#c3c2b7"
SURFACE: str = "#fcfcfb"


def apply_style() -> None:
    """Recessive chrome so the data leads, per the viz palette."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SOFT,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_features() -> pd.DataFrame:
    """Load ESC-10 and measure loudness, brightness, and noisiness per clip."""
    meta = pd.read_csv(RAW_DIR / "meta" / "esc50.csv")
    esc10 = meta[meta["esc10"]].reset_index(drop=True)
    audio_dir = RAW_DIR / "audio"

    rows: List[Dict[str, object]] = []
    for _, r in esc10.iterrows():
        waveform, _ = librosa.load(audio_dir / r["filename"], sr=SAMPLE_RATE, duration=DURATION)
        rows.append({
            "category": r["category"],
            "filename": r["filename"],
            "rms": float(np.mean(librosa.feature.rms(y=waveform))),
            "centroid": float(np.mean(librosa.feature.spectral_centroid(y=waveform, sr=SAMPLE_RATE))),
        })
    return pd.DataFrame(rows)


def chart_class_distribution(df: pd.DataFrame) -> None:
    counts = df["category"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(counts.index, counts.values, color=BLUE, width=0.7)
    ax.bar_label(bars, color=INK_SOFT, padding=3, fontsize=9)
    ax.set_ylabel("Clips")
    ax.set_title("Class distribution: perfectly balanced")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "class_distribution.png", dpi=120)
    plt.close(fig)


def sorted_boxplot(df: pd.DataFrame, column: str, ylabel: str, title: str, filename: str) -> None:
    order = df.groupby("category")[column].median().sort_values().index.tolist()
    data = [df[df["category"] == c][column].values for c in order]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    box = ax.boxplot(data, labels=order, patch_artist=True, medianprops={"color": INK})
    for patch in box["boxes"]:
        patch.set_facecolor(BLUE)
        patch.set_alpha(0.75)
        patch.set_edgecolor(AXIS)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / filename, dpi=120)
    plt.close(fig)


def chart_spectrograms(df: pd.DataFrame) -> None:
    audio_dir = RAW_DIR / "audio"
    sample = df.groupby("category").first().reset_index()
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for ax, (_, row) in zip(axes.ravel(), sample.iterrows()):
        waveform, _ = librosa.load(audio_dir / row["filename"], sr=SAMPLE_RATE, duration=DURATION)
        mel = librosa.feature.melspectrogram(y=waveform, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        ax.imshow(log_mel, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(row["category"], fontsize=9, color=INK)
        ax.axis("off")
    fig.suptitle("Mel-spectrograms the CNN learns from", color=INK)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "melspectrograms.png", dpi=120)
    plt.close(fig)


def write_insights(df: pd.DataFrame) -> None:
    loudest = df.groupby("category")["rms"].median().idxmax()
    quietest = df.groupby("category")["rms"].median().idxmin()
    brightest = df.groupby("category")["centroid"].median().idxmax()
    darkest = df.groupby("category")["centroid"].median().idxmin()

    insights: Dict[str, object] = {
        "num_clips": int(len(df)),
        "num_classes": int(df["category"].nunique()),
        "clips_per_class": int(df["category"].value_counts().iloc[0]),
        "features": [
            {
                "name": "Class distribution",
                "chart": "class_distribution.png",
                "story": "Every class has the same number of clips, so the dataset is perfectly balanced. Plain accuracy is a fair headline metric and no class weighting is needed.",
            },
            {
                "name": "Loudness (RMS energy)",
                "chart": "loudness_by_class.png",
                "story": f"Sustained sources carry high, steady energy while transient sounds sit low. Here '{loudest}' is loudest and '{quietest}' is quietest, so energy alone already separates several classes.",
            },
            {
                "name": "Brightness (spectral centroid)",
                "chart": "brightness_by_class.png",
                "story": f"Spectral centroid marks where the spectrum's center of mass sits. '{brightest}' skews high and bright while '{darkest}' skews low, giving the model a cue that complements loudness.",
            },
        ],
    }
    (ASSETS_DIR / "insights.json").write_text(json.dumps(insights, indent=2))


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()
    df = load_features()
    chart_class_distribution(df)
    sorted_boxplot(df, "rms", "Mean RMS energy", "Loudness by class", "loudness_by_class.png")
    sorted_boxplot(df, "centroid", "Mean spectral centroid (Hz)", "Brightness by class", "brightness_by_class.png")
    chart_spectrograms(df)
    write_insights(df)
    print("Wrote insight assets to", ASSETS_DIR)


if __name__ == "__main__":
    main()
