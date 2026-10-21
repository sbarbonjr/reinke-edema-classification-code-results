#!/usr/bin/env python3
"""
prepare_dataset.py (windowed / segmented + WPD Haar-7 + optional wav2vec2/wavlm with auto-resample)

Extract MANY segments per WAV (hundreds) while avoiding leakage via:
- group_id = original WAV stem (all segments from same WAV share the same group_id)
- segment_id = unique per segment within the file

Folder structure expected:
  wav-files/
    Portuguese_Healthy/*.wav
    Portuguese_Reinke/*.wav

Generates CSVs (each row = one SEGMENT):
  - manifest.csv
  - mfcc_stats.csv           (optional)
  - logmel_stats.csv         (optional)
  - spectral_stats.csv       (optional)
  - fft_topk_<window>.csv    (optional)
  - wpd_haar7_std.csv        (optional)  <-- paper-aligned features
  - wav2vec2.csv             (optional, expensive)

IMPORTANT for leakage-free experiments:
- Use group-aware CV (GroupKFold / StratifiedGroupKFold) with group_id
- Do feature selection INSIDE the CV pipeline, not before

Paper-aligned WPD setup:
- Resample to 22050 Hz
- Segment into non-overlapping windows of 4096 samples:
    segment_sec = hop_sec = 4096 / 22050 ≈ 0.18576
- WPD Haar, level=7, feature = std of leaf coefficients => 128 features

Embeddings:
- You can run the whole pipeline at --sr 22050 (for WPD), and embeddings will be computed
  by resampling each segment internally to 16k before feeding the HF model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator

import numpy as np
import pandas as pd

try:
    import librosa
except Exception as e:
    raise RuntimeError("librosa is required. Install with: pip install librosa soundfile") from e

try:
    import pywt
except Exception as e:
    raise RuntimeError("PyWavelets is required for --wpd. Install with: pip install PyWavelets") from e


META_COLS = [
    "file", "filename", "group_id",
    "segment_id", "start_sample", "start_sec",
    "label", "class_name", "sr", "segment_sec"
]


# ----------------------------
# Utilities
# ----------------------------

def safe_stem(p: Path) -> str:
    return p.stem


def list_wavs(
    data_root: Path,
    healthy_dirname: str = "Portuguese_Healthy",
    reinke_dirname: str = "Portuguese_Reinke"
) -> List[Tuple[Path, int]]:
    """Returns list of (wav_path, label), where label: 0=Healthy, 1=Reinke."""
    healthy_dir = data_root / healthy_dirname
    reinke_dir = data_root / reinke_dirname

    if not healthy_dir.exists():
        raise FileNotFoundError(f"Healthy folder not found: {healthy_dir}")
    if not reinke_dir.exists():
        raise FileNotFoundError(f"Reinke folder not found: {reinke_dir}")

    healthy = sorted([p for p in healthy_dir.glob("*.wav") if p.is_file()])
    reinke = sorted([p for p in reinke_dir.glob("*.wav") if p.is_file()])

    files: List[Tuple[Path, int]] = [(p, 0) for p in healthy] + [(p, 1) for p in reinke]
    if not files:
        raise RuntimeError("No .wav files found.")
    return files


def load_audio(path: Path, sr: int, mono: bool = True) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=mono)
    if y is None or len(y) == 0:
        raise RuntimeError(f"Empty audio: {path}")
    return y.astype(np.float32)


def trim_silence(y: np.ndarray, top_db: int = 30) -> np.ndarray:
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    return yt if len(yt) > 0 else y


def rms_normalize(y: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    rms = np.sqrt(np.mean(y * y) + eps)
    return y / rms


def stat_pool(feat: np.ndarray, prefix: str, include_mean: bool = True, include_std: bool = True) -> Dict[str, float]:
    """
    feat: (n_features, n_frames) or (n_frames,) -> pool along frames dimension.
    Returns dict of pooled stats.
    """
    if feat.ndim == 1:
        feat2 = feat[None, :]
    else:
        feat2 = feat

    out: Dict[str, float] = {}
    if include_mean:
        m = np.mean(feat2, axis=1)
        for i, v in enumerate(m):
            out[f"{prefix}_mean_{i:02d}"] = float(v)
    if include_std:
        s = np.std(feat2, axis=1)
        for i, v in enumerate(s):
            out[f"{prefix}_std_{i:02d}"] = float(v)
    return out


# ----------------------------
# Segmentation
# ----------------------------

def iter_segments(
    y: np.ndarray,
    sr: int,
    segment_sec: float,
    hop_sec: float,
    pad_last: bool = True,
    max_segments: Optional[int] = None
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """Yields (seg_idx, start_sample, segment_audio)."""
    seg_len = int(round(segment_sec * sr))
    hop_len = int(round(hop_sec * sr))

    if seg_len <= 0 or hop_len <= 0:
        raise ValueError("segment_sec and hop_sec must be > 0")
    if len(y) == 0:
        return

    seg_idx = 0
    start = 0
    while start < len(y):
        end = start + seg_len
        seg = y[start:end]
        if len(seg) < seg_len:
            if not pad_last:
                break
            seg = np.pad(seg, (0, seg_len - len(seg)), mode="constant")

        yield seg_idx, start, seg.astype(np.float32)

        seg_idx += 1
        if max_segments is not None and seg_idx >= max_segments:
            break
        start += hop_len


# ----------------------------
# Feature extractors
# ----------------------------

def extract_mfcc_stats(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 1024,
    hop_length: int = 256,
    include_deltas: bool = True
) -> Dict[str, float]:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    feats: Dict[str, float] = {}
    feats.update(stat_pool(mfcc, "mfcc"))
    if include_deltas:
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)
        feats.update(stat_pool(d1, "mfcc_d1"))
        feats.update(stat_pool(d2, "mfcc_d2"))
    return feats


def extract_logmel_stats(
    y: np.ndarray,
    sr: int,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 256
) -> Dict[str, float]:
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0)
    logmel = librosa.power_to_db(mel + 1e-10)
    return stat_pool(logmel, "logmel")


def extract_spectral_stats(y: np.ndarray, sr: int, n_fft: int = 1024, hop_length: int = 256) -> Dict[str, float]:
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)

    feats: Dict[str, float] = {}
    feats.update(stat_pool(zcr.squeeze(), "zcr"))
    feats.update(stat_pool(rms.squeeze(), "rms"))
    feats.update(stat_pool(centroid.squeeze(), "centroid"))
    feats.update(stat_pool(bandwidth.squeeze(), "bandwidth"))
    feats.update(stat_pool(rolloff.squeeze(), "rolloff"))
    feats.update(stat_pool(flatness.squeeze(), "flatness"))
    return feats


def extract_fft_topk(
    y: np.ndarray,
    sr: int,
    n_fft: int = 4096,
    k_top: int = 64,
    window: str = "hann"
) -> Dict[str, float]:
    """
    Global FFT magnitude for a segment:
      - pad/crop to n_fft
      - apply window with same length
      - rFFT
      - keep top-k magnitudes and their frequencies
    """
    y2 = y.astype(np.float32)
    if len(y2) < n_fft:
        y2 = np.pad(y2, (0, n_fft - len(y2)), mode="constant")
    elif len(y2) > n_fft:
        y2 = y2[:n_fft]

    if window == "hann":
        w = np.hanning(len(y2)).astype(np.float32)
    elif window == "hamming":
        w = np.hamming(len(y2)).astype(np.float32)
    elif window == "rect":
        w = np.ones(len(y2), dtype=np.float32)
    else:
        raise ValueError("window must be one of: hann, hamming, rect")

    y2 = y2 * w
    spec = np.fft.rfft(y2, n=len(y2))
    mag = np.abs(spec).astype(np.float32)

    if len(mag) > 0:
        mag[0] = 0.0

    idx = np.argsort(mag)[::-1][:k_top]
    top_mag = mag[idx]
    freqs = (idx * (sr / len(y2))).astype(np.float32)

    feats: Dict[str, float] = {}
    for i in range(len(idx)):
        feats[f"fft_top_mag_{i:03d}"] = float(top_mag[i])
        feats[f"fft_top_hz_{i:03d}"] = float(freqs[i])
    return feats


def extract_wpd_haar_std(y: np.ndarray, level: int = 7, mode: str = "symmetric") -> Dict[str, float]:
    """
    Wavelet Packet Decomposition (Haar), `level` levels.
    Returns std of each leaf node coefficient vector.
    Leaves = 2^level (level=7 => 128 features).
    """
    wp = pywt.WaveletPacket(data=y, wavelet="haar", mode=mode, maxlevel=level)
    nodes = wp.get_level(level, order="freq")
    stds = [float(np.std(n.data)) for n in nodes]

    target = 2 ** level
    if len(stds) != target:
        if len(stds) < target:
            stds = stds + [0.0] * (target - len(stds))
        else:
            stds = stds[:target]

    return {f"wpd_haar_l{level}_std_{i:03d}": stds[i] for i in range(len(stds))}


# ----------------------------
# Optional embeddings (wav2vec2 / wavlm) with AUTO RESAMPLE to 16k
# ----------------------------

def can_use_wav2vec2() -> bool:
    try:
        import torch  # noqa: F401
        from transformers import AutoFeatureExtractor, AutoModel  # noqa: F401
        return True
    except Exception:
        return False


class Wav2Vec2Embedder:
    """
    Frozen embeddings for wav2vec2-like models.
    Uses AutoFeatureExtractor (audio-only).
    - If the input segment sample rate != 16000, it is resampled internally to 16000.

    Works with:
    - ASR/CTC checkpoints (e.g., facebook/wav2vec2-base-960h, microsoft/wavlm-base):
        loads AutoModelForCTC then uses .wav2vec2 / .wavlm encoder when available.
    - Representation checkpoints:
        loads AutoModel.
    """
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h", device: Optional[str] = None):
        import torch
        from transformers import AutoConfig, AutoModel, AutoModelForCTC, AutoFeatureExtractor

        self.torch = torch
        self.feat_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        cfg = AutoConfig.from_pretrained(model_name)

        encoder = None
        try:
            ctc = AutoModelForCTC.from_pretrained(model_name, config=cfg)
            # wav2vec2-based models often expose .wav2vec2
            encoder = getattr(ctc, "wav2vec2", None)
            # wavlm models often expose .wavlm
            if encoder is None:
                encoder = getattr(ctc, "wavlm", None)
        except Exception:
            encoder = None

        if encoder is None:
            encoder = AutoModel.from_pretrained(model_name, config=cfg)

        self.model = encoder

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def embed(self, y: np.ndarray, sr: int) -> np.ndarray:
        # OPTION B: auto-resample to 16k
        if sr != 16000:
            y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000)
            sr = 16000

        torch = self.torch
        inputs = self.feat_extractor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model(**inputs)
            h = out.last_hidden_state  # (B,T,D)
            pooled = h.mean(dim=1)     # (B,D)
            return pooled.squeeze(0).detach().cpu().numpy().astype(np.float32)


# ----------------------------
# Main pipeline
# ----------------------------

def build_rows(
    files: List[Tuple[Path, int]],
    sr: int,
    do_trim: bool,
    do_rms_norm: bool,
    trim_top_db: int,
    segment_sec: float,
    hop_sec: float,
    pad_last: bool,
    max_segments_per_file: Optional[int],
    min_file_sec: float,
    mfcc: bool,
    logmel: bool,
    spectral: bool,
    fft: bool,
    fft_k: int,
    fft_window: str,
    wpd: bool,
    wav2vec2: bool,
    wav2vec2_model: str,
    cache_dir: Optional[Path],
) -> Dict[str, List[Dict[str, object]]]:

    rows: Dict[str, List[Dict[str, object]]] = {"manifest": []}
    if mfcc:
        rows["mfcc_stats"] = []
    if logmel:
        rows["logmel_stats"] = []
    if spectral:
        rows["spectral_stats"] = []
    if fft:
        rows[f"fft_topk_{fft_window}"] = []
    if wpd:
        rows["wpd_haar7_std"] = []
    if wav2vec2:
        rows["wav2vec2"] = []

    embedder = None
    if wav2vec2:
        if not can_use_wav2vec2():
            print("[WARN] wav2vec2 requested but transformers/torch not available. Skipping embeddings.")
            wav2vec2 = False
        else:
            embedder = Wav2Vec2Embedder(model_name=wav2vec2_model)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    seg_len = int(round(segment_sec * sr))
    if seg_len <= 0:
        raise ValueError("segment_sec too small for the selected sr.")

    total_segments = 0

    for wav_path, label in files:
        y = load_audio(wav_path, sr=sr, mono=True)

        if do_trim:
            y = trim_silence(y, top_db=trim_top_db)
        if do_rms_norm:
            y = rms_normalize(y)

        file_sec = len(y) / float(sr)
        if file_sec < min_file_sec:
            print(f"[SKIP] {wav_path.name} duration={file_sec:.3f}s < min_file_sec={min_file_sec:.3f}s")
            continue

        group_id = safe_stem(wav_path)

        seg_count_this_file = 0
        for seg_idx, start_sample, seg in iter_segments(
            y=y,
            sr=sr,
            segment_sec=segment_sec,
            hop_sec=hop_sec,
            pad_last=pad_last,
            max_segments=max_segments_per_file,
        ):
            seg_count_this_file += 1
            total_segments += 1

            meta = {
                "file": str(wav_path),
                "filename": wav_path.name,
                "group_id": group_id,
                "segment_id": int(seg_idx),
                "start_sample": int(start_sample),
                "start_sec": float(start_sample / sr),
                "segment_sec": float(segment_sec),
                "sr": int(sr),
                "label": int(label),
                "class_name": "Healthy" if label == 0 else "Reinke",
            }
            rows["manifest"].append(dict(meta))

            if mfcc:
                feats = extract_mfcc_stats(seg, sr=sr, n_mfcc=13, include_deltas=True)
                rows["mfcc_stats"].append({**meta, **feats})

            if logmel:
                feats = extract_logmel_stats(seg, sr=sr, n_mels=64)
                rows["logmel_stats"].append({**meta, **feats})

            if spectral:
                feats = extract_spectral_stats(seg, sr=sr)
                rows["spectral_stats"].append({**meta, **feats})

            if fft:
                feats = extract_fft_topk(seg, sr=sr, n_fft=4096, k_top=fft_k, window=fft_window)
                rows[f"fft_topk_{fft_window}"].append({**meta, **feats})

            if wpd:
                feats = extract_wpd_haar_std(seg, level=7)
                rows["wpd_haar7_std"].append({**meta, **feats})

            if wav2vec2 and embedder is not None:
                emb_key = (
                    f"{wav_path.stem}"
                    f"_seg{seg_idx:05d}"
                    f"_sr{sr}"
                    f"_segsec{segment_sec}"
                    f"_{wav2vec2_model.replace('/', '_')}.npy"
                )
                emb_path = (cache_dir / emb_key) if cache_dir is not None else None

                if emb_path is not None and emb_path.exists():
                    emb = np.load(emb_path).astype(np.float32)
                else:
                    emb = embedder.embed(seg, sr=sr)
                    if emb_path is not None:
                        np.save(emb_path, emb)

                emb_feats = {f"emb_{i:04d}": float(v) for i, v in enumerate(emb.tolist())}
                rows["wav2vec2"].append({**meta, **emb_feats})

        print(f"[INFO] {wav_path.name}: segments_extracted={seg_count_this_file}")

    print(f"[INFO] Total segments extracted: {total_segments}")
    return rows


def save_csvs(rows: Dict[str, List[Dict[str, object]]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, rlist in rows.items():
        df = pd.DataFrame(rlist)

        cols = [c for c in META_COLS if c in df.columns] + [c for c in df.columns if c not in META_COLS]
        df = df[cols]

        out_path = out_dir / f"{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"[OK] Wrote {out_path}  shape={df.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True,
                    help="Path containing Portuguese_Healthy and Portuguese_Reinke folders.")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output directory for CSV feature tables.")
    ap.add_argument("--healthy_dirname", type=str, default="Portuguese_Healthy")
    ap.add_argument("--reinke_dirname", type=str, default="Portuguese_Reinke")

    # Audio preprocessing
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--trim_silence", action="store_true")
    ap.add_argument("--trim_top_db", type=int, default=30)
    ap.add_argument("--rms_norm", action="store_true")

    # Segmentation controls
    ap.add_argument("--segment_sec", type=float, default=0.25,
                    help="Length of each extracted segment in seconds.")
    ap.add_argument("--hop_sec", type=float, default=0.05,
                    help="Hop between segments in seconds.")
    ap.add_argument("--pad_last", action="store_true",
                    help="Pad last segment if it is shorter than segment_sec.")
    ap.add_argument("--max_segments_per_file", type=int, default=None,
                    help="Optional cap on number of segments per file.")
    ap.add_argument("--min_file_sec", type=float, default=0.0,
                    help="Skip files shorter than this duration after trimming.")

    # Feature toggles
    ap.add_argument("--mfcc", action="store_true")
    ap.add_argument("--logmel", action="store_true")
    ap.add_argument("--spectral", action="store_true")
    ap.add_argument("--fft", action="store_true")
    ap.add_argument("--fft_k", type=int, default=64)
    ap.add_argument("--fft_window", type=str, default="hann", choices=["hann", "hamming", "rect"])
    ap.add_argument("--wpd", action="store_true",
                    help="Extract WPD Haar level-7 std (128 feats) -> wpd_haar7_std.csv")

    # Embeddings
    ap.add_argument("--wav2vec2", action="store_true")
    ap.add_argument("--wav2vec2_model", type=str, default="facebook/wav2vec2-base-960h")
    ap.add_argument("--cache_dir", type=str, default=None,
                    help="Optional directory to cache embeddings as .npy (per segment).")

    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    files = list_wavs(data_root, args.healthy_dirname, args.reinke_dirname)
    print(f"[INFO] Found {len(files)} wav files: "
          f"{sum(1 for _, y in files if y == 0)} Healthy, {sum(1 for _, y in files if y == 1)} Reinke")

    # Default features only if user didn't request any feature set
    if not (args.mfcc or args.logmel or args.spectral or args.fft or args.wav2vec2 or args.wpd):
        print("[INFO] No feature flags set. Enabling default: --mfcc --spectral --fft")
        args.mfcc = True
        args.spectral = True
        args.fft = True

    rows = build_rows(
        files=files,
        sr=args.sr,
        do_trim=args.trim_silence,
        do_rms_norm=args.rms_norm,
        trim_top_db=args.trim_top_db,
        segment_sec=args.segment_sec,
        hop_sec=args.hop_sec,
        pad_last=args.pad_last,
        max_segments_per_file=args.max_segments_per_file,
        min_file_sec=args.min_file_sec,
        mfcc=args.mfcc,
        logmel=args.logmel,
        spectral=args.spectral,
        fft=args.fft,
        fft_k=args.fft_k,
        fft_window=args.fft_window,
        wpd=args.wpd,
        wav2vec2=args.wav2vec2,
        wav2vec2_model=args.wav2vec2_model,
        cache_dir=cache_dir,
    )

    save_csvs(rows, out_dir)


if __name__ == "__main__":
    main()