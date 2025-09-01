import os, sys, argparse, uuid, ctypes, json
from pathlib import Path
import numpy as np
import tempfile, gc, traceback
import soundfile as sf

# ---- Windows CUDA DLLs (safe to ignore if not present)
if sys.platform == "win32":
    for ver in ("v12.4","v12.3","v12.2","v12.1","v12.0"):
        b = fr"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{ver}\bin"
        if os.path.isdir(b):
            try: os.add_dll_directory(b)
            except Exception: pass
    # Optional: preload cublas if available
    dll = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\cublas64_12.dll"
    if os.path.exists(dll):
        try: ctypes.WinDLL(dll)
        except OSError: pass

import soundfile as sf
try:
    import librosa
    HAVE_LIBROSA = True
except Exception:
    HAVE_LIBROSA = False

try:
    from scipy.signal import resample_poly
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

import webrtcvad
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from faster_whisper import WhisperModel

# ------------- helpers -------------
def resample_to(y, sr_in, sr_out):
    """Resample y from sr_in -> sr_out using the best available backend."""
    if sr_in == sr_out:
        return y.astype(np.float32, copy=False)

    # Prefer SciPy's resample_poly (fast + good quality)
    try:
        from scipy.signal import resample_poly  # local import is fine (not assigning to `np`)
        from math import gcd
        g = gcd(sr_in, sr_out)
        up, down = sr_out // g, sr_in // g
        y = resample_poly(y, up, down)
        return y.astype(np.float32, copy=False)
    except Exception:
        pass

    # Next, try librosa
    try:
        import librosa
        y = librosa.resample(y, orig_sr=sr_in, target_sr=sr_out)
        return y.astype(np.float32, copy=False)
    except Exception:
        pass

    # Fallback: pure NumPy linear interpolation (slower but safe)
    n_out = int(round(len(y) * sr_out / float(sr_in)))
    if n_out <= 1:
        return y.astype(np.float32, copy=False)
    t_old = np.linspace(0.0, len(y) / sr_in, num=len(y), endpoint=False, dtype=np.float64)
    t_new = np.linspace(0.0, len(y) / sr_in, num=n_out, endpoint=False, dtype=np.float64)
    y = np.interp(t_new, t_old, y)
    return y.astype(np.float32, copy=False)

def iter_wav_chunks(path, chunk_sec=600.0, overlap_sec=0.5, target_sr=16000, debug=False):
    """
    Stream a (possibly huge) audio file in chunks. Works for WAV/FLAC/AIFF;
    if MP3 fails via libsndfile, convert to WAV externally first.
    Yields (y16k_mono, target_sr, chunk_index, chunk_start_sec).
    """
    try:
        with sf.SoundFile(str(path), 'r') as f:
            sr = f.samplerate
            ch = f.channels
            total_frames = len(f)
            if debug:
                print(f"[open] {path.name}: sr={sr}, ch={ch}, frames={total_frames}")

            chunk_frames = int(chunk_sec * sr)
            hop_frames = max(1, int((chunk_sec - overlap_sec) * sr))
            i = 0
            pos = 0
            while pos < total_frames:
                n = min(chunk_frames, total_frames - pos)
                f.seek(pos)
                # read as float32, 2D if multichannel
                y = f.read(n, dtype="float32", always_2d=True)  # shape (n, ch)
                if y.size == 0:
                    break
                # mono
                if y.shape[1] > 1:
                    y = y.mean(axis=1)
                else:
                    y = y[:,0]
                # resample to target_sr
                y = resample_to(y, sr, target_sr)
                # normalize per chunk (avoid clipping)
                peak = float(np.max(np.abs(y))) if y.size else 0.0
                if peak > 1e-6:
                    y = 0.97 * (y / peak)
                y = y.astype(np.float32)
                if debug:
                    print(f"  [chunk] idx={i} pos={pos} frames_read={n} -> len16k={len(y)}")
                yield y, target_sr, i, pos / sr
                i += 1
                pos += hop_frames
    except Exception as e:
        raise RuntimeError(f"iter_wav_chunks failed for {path.name}: {e}")

def framed_pcm16(y16k, frame_ms=30):
    sr = 16000
    frame_len = int(sr * (frame_ms/1000.0))
    # step = frame_len (non-overlap) is OK for webrtcvad
    for i in range(0, len(y16k) - frame_len + 1, frame_len):
        frame = y16k[i:i+frame_len]
        pcm16 = (np.clip(frame, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        yield pcm16

def vad_segments(y16k, aggressiveness=2, min_sec=1.2, max_sec=12.0, hang_ms=300):
    vad = webrtcvad.Vad(aggressiveness)
    sr = 16000
    frame_ms = 30
    frame_len = int(sr * (frame_ms/1000.0))
    hang_frames = int(hang_ms / frame_ms)

    in_seg = False
    start_f = 0
    hang = 0
    segs = []

    for idx, b in enumerate(framed_pcm16(y16k, frame_ms)):
        v = vad.is_speech(b, sr)
        if v:
            if not in_seg:
                in_seg = True
                start_f = idx
            hang = 0
        else:
            if in_seg:
                hang += 1
                if hang > hang_frames:
                    end_f = max(start_f, idx - hang)
                    a = start_f * frame_len
                    b = (end_f + 1) * frame_len
                    dur = (b - a) / sr
                    if min_sec <= dur <= max_sec:
                        segs.append((a, b))
                    in_seg = False
                    hang = 0
    if in_seg:
        a = start_f * frame_len
        b = len(y16k)
        dur = (b - a) / sr
        if min_sec <= dur <= max_sec:
            segs.append((a, b))
    return segs

def save_wav_22k(y16k, out_path):
    y22 = resample_to(y16k, 16000, 22050)
    peak = float(np.max(np.abs(y22))) if y22.size else 0.0
    if peak > 1e-6:
        y22 = 0.97 * (y22 / peak)
    sf.write(str(out_path), y22, 22050, subtype="PCM_16")

def sanitize_wav(src_path, dst_path, target_sr=16000):
    """Re-wrap audio to a clean mono PCM16 WAV at target_sr, removing NaNs/Inf and DC."""
    y, sr = sf.read(str(src_path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    # remove NaN/Inf and DC
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = y - float(np.mean(y)) if y.size else y
    # resample
    y = resample_to(y, sr, target_sr).astype(np.float32)
    # normalize a bit
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-6:
        y = 0.97 * (y / peak)
    sf.write(str(dst_path), y, target_sr, subtype="PCM_16")

def split_wav(src_path, target_sr=16000):
    """Return two temp files that split the input in half (both repaired to mono/16k/PCM16)."""
    y, sr = sf.read(str(src_path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if y.size < target_sr:   # <1s, don't split further
        return []
    mid = y.size // 2
    parts = [(y[:mid], sr), (y[mid:], sr)]
    outs = []
    for arr, s in parts:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            arr = resample_to(arr, s, target_sr)
            peak = float(np.max(np.abs(arr))) if arr.size else 0.0
            if peak > 1e-6:
                arr = 0.97 * (arr / peak)
            sf.write(tmp.name, arr, target_sr, subtype="PCM_16")
            outs.append(tmp.name)
    return outs

def transcribe_paths(seg_paths, lang_hint=None, model_size="medium", device=None, compute_type=None, debug=False):
    from faster_whisper import WhisperModel
    use_cuda = torch.cuda.is_available() if device is None else (device == "cuda")
    prim_device = "cuda" if use_cuda else "cpu"
    prim_compute = compute_type or ("float16" if prim_device == "cuda" else "int8")

    if debug:
        print(f"[whisper:init] device={prim_device} compute_type={prim_compute} model={model_size} segments={len(seg_paths)}")

    bad_list_path = Path("bad_segments.txt")
    bad_log = bad_list_path.open("a", encoding="utf-8")

    def new_model(dev, ctype):
        if debug: print(f"[whisper:new] device={dev} compute_type={ctype}")
        return WhisperModel(model_size, device=dev, compute_type=ctype)

    def decode_path(wav_path, model, label):
        segments, info = model.transcribe(
            str(wav_path),
            language=lang_hint,
            beam_size=5,
            vad_filter=False,
            word_timestamps=False,
            # safer options if you still see flakes:
            # temperature=0.0, best_of=1, patience=0
        )
        txt = " ".join([(s.text or "").strip() for s in segments if (s.text or "").strip()]).strip()
        return txt, getattr(info, "language", None)

    model = new_model(prim_device, prim_compute)
    rows = []

    for i, p in enumerate(seg_paths, 1):
        tried = []
        def try_decode(path_str, dev, ctype, label):
            nonlocal model
            tried.append(label)
            try:
                if (dev != prim_device) or (ctype != prim_compute):
                    # build fresh model on change of backend
                    del model; gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    model = new_model(dev, ctype)
                txt, lang = decode_path(path_str, model, label)
                return txt, lang
            except Exception as e:
                if debug:
                    print(f"[asr-error:{label}] {Path(path_str).name}: {e}")
                    print(traceback.format_exc())
                return None, None

        # 1) primary attempt
        txt, lang = try_decode(str(p), prim_device, prim_compute, f"{prim_device}[{prim_compute}]")
        if not txt:
            # 2) GPU safer quant
            if prim_device == "cuda":
                txt, lang = try_decode(str(p), "cuda", "int8_float16", "cuda[int8_float16]")
            # 3) CPU int8
            if not txt:
                txt, lang = try_decode(str(p), "cpu", "int8", "cpu[int8]")

        # 4) sanitize+retry (CPU) if still empty
        if not txt:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                try:
                    sanitize_wav(p, tmp.name, target_sr=16000)
                    txt, lang = try_decode(tmp.name, "cpu", "int8", "cpu[int8]+sanitize")
                except Exception as e:
                    if debug: print(f"[sanitize-fail] {p.name}: {e}")

        # 5) split+retry (CPU) if still empty
        if not txt:
            parts = split_wav(p, target_sr=16000)
            if parts:
                texts = []
                for idx, part in enumerate(parts):
                    t, _ = try_decode(part, "cpu", "int8", f"cpu[int8]+split{idx}")
                    if t: texts.append(t)
                if texts:
                    txt, lang = " ".join(texts), (lang_hint or "auto")

        if txt:
            rows.append((p.stem, txt, lang_hint or (lang or "auto")))
        else:
            # give up on this one
            bad_log.write(f"{p}\n")
            if debug:
                print(f"[asr-skip] {p.name} after retries: {tried}")

        # periodically trim memory
        if (i % 10) == 0 and torch.cuda.is_available():
            try: torch.cuda.empty_cache()
            except Exception: pass

    bad_log.close()
    return rows

# ------------- main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True, help="Folder containing audio files (WAV/FLAC/etc).")
    ap.add_argument("--out_dir", default="data/processed_auto", help="Output dir (LJSpeech-style).")
    ap.add_argument("--lang", default=None, help="Language hint like en/ja/zh/zh-cn; default: autodetect.")
    ap.add_argument("--model_size", default="medium", help="faster-whisper size: tiny/small/medium/large-v3.")
    ap.add_argument("--device", default=None, choices=[None,"cpu","cuda"], help="Force device; default: auto.")
    ap.add_argument("--compute_type", default=None, help="Whisper compute type (e.g., int8, int8_float16, float16).")

    # Chunking + VAD
    ap.add_argument("--pre_chunk_sec", type=float, default=600.0, help="Chunk length in seconds (default 10 min).")
    ap.add_argument("--pre_chunk_overlap_sec", type=float, default=0.5, help="Overlap between chunks (sec).")
    ap.add_argument("--vad_aggr", type=int, default=2, help="WebRTC VAD aggressiveness 0-3.")
    ap.add_argument("--min_sec", type=float, default=1.2, help="Minimum segment length.")
    ap.add_argument("--max_sec", type=float, default=12.0, help="Maximum segment length.")

    # Safety caps
    ap.add_argument("--max_segments_per_file", type=int, default=2000, help="Cap segments per source file.")
    ap.add_argument("--max_segments_total", type=int, default=20000, help="Global cap across all files.")

    ap.add_argument("--debug", action="store_true", help="Verbose logging.")
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir = out_dir / "wavs"; wavs_dir.mkdir(parents=True, exist_ok=True)

    seg_paths = []
    total_count = 0

    for src in sorted(audio_dir.iterdir()):
        if src.suffix.lower() not in [".wav", ".flac", ".aiff", ".aif", ".mp3", ".m4a", ".ogg"]:
            continue
        if args.debug:
            print(f"[file] {src.name}")

        produced = 0
        try:
            for y16k, _, chunk_idx, start_sec in iter_wav_chunks(
                src, chunk_sec=args.pre_chunk_sec, overlap_sec=args.pre_chunk_overlap_sec, target_sr=16000, debug=args.debug
            ):
                segs = vad_segments(
                    y16k,
                    aggressiveness=args.vad_aggr,
                    min_sec=args.min_sec,
                    max_sec=args.max_sec,
                    hang_ms=300,
                )
                if args.debug:
                    print(f"    [vad] chunk={chunk_idx} start={start_sec:.1f}s segs={len(segs)}")

                for (a, b) in segs:
                    seg = y16k[a:b]
                    utt = uuid.uuid4().hex[:12]
                    out_wav = wavs_dir / f"{utt}.wav"
                    save_wav_22k(seg, out_wav)
                    seg_paths.append(out_wav)
                    produced += 1
                    total_count += 1

                    if produced >= args.max_segments_per_file:
                        if args.debug:
                            print(f"    [cap] per-file {src.name} reached {args.max_segments_per_file}")
                        break
                    if total_count >= args.max_segments_total:
                        if args.debug:
                            print(f"[cap] global reached {args.max_segments_total}")
                        break

                # free CUDA/CPU cache between chunks
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                if produced >= args.max_segments_per_file or total_count >= args.max_segments_total:
                    break

        except Exception as e:
            print(f"[seg-skip] {src.name}: {e}")
            continue

        if args.debug:
            print(f"  [file-done] {src.name} -> segments={produced}")

        if total_count >= args.max_segments_total:
            break

    if not seg_paths:
        print("[stop] No segments produced. Try: lower --min_sec, lower VAD aggressiveness, or increase --pre_chunk_sec.")
        sys.exit(1)

    if args.debug:
        print(f"[asr] total segments: {len(seg_paths)}")

    rows = transcribe_paths(
        seg_paths,
        lang_hint=args.lang,
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        debug=args.debug,
    )
    if not rows:
        print("[stop] No transcripts produced. Try a larger ASR model (e.g. --model_size large-v3) or set --device cpu --compute_type int8.")
        sys.exit(1)

    df = pd.DataFrame(rows, columns=["utt", "text", "lang"])

    # 10% val if small set, else 5%
    val_size = 0.1 if len(df) <= 200 else 0.05
    train_ids, val_ids = train_test_split(df["utt"], test_size=val_size, random_state=42)

    with open(out_dir / "metadata.csv", "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(f"{r['utt']}|{r['text']}|{r['text']}\n")
    with open(out_dir / "metadata_val.csv", "w", encoding="utf-8") as f:
        for _, r in df[df["utt"].isin(val_ids)].iterrows():
            f.write(f"{r['utt']}|{r['text']}|{r['text']}\n")

    print(f"[done] Segments: {len(df)} | wrote: {out_dir}/metadata.csv")


if __name__ == "__main__":
    main()