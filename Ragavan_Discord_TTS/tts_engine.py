# tts_engine.py
"""
Minimal TTS wrapper for Discord bot (Coqui XTTS v2) with PyTorch 2.6+
safe-globals allow-list for XTTS checkpoints.

- Zero/low-shot voice cloning via reference WAVs (speaker_wav list)
- CPU/GPU supported; GPU recommended
"""

import inspect
import os
import uuid
from typing import List, Optional, Union
import importlib

import numpy as np
import torch
import torchaudio


def _import_from_string(dotted_path: str):
    """
    Import an object from a dotted path string. Returns the object or None.
    """
    try:
        module_path, obj_name = dotted_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        return getattr(mod, obj_name, None)
    except Exception:
        return None


def _register_safe_globals():
    """
    On PyTorch 2.6+, torch.load defaults to weights_only=True and blocks unpickling
    of arbitrary classes. XTTS checkpoints reference several XTTS config classes.
    We allow-list the ones commonly encountered.

    This is safe if you trust the TTS package and its checkpoints.
    """
    try:
        from torch.serialization import add_safe_globals  # PyTorch 2.6+
    except Exception:
        return  # Older PyTorch; nothing to do.

    candidates = [
        # Seen in errors and often used by XTTS:
        "TTS.tts.configs.xtts_config.XttsConfig",
        "TTS.tts.models.xtts.XttsAudioConfig",
        # Extra likely configs/types to pre-empt further errors (best-effort):
        "TTS.tts.configs.shared_configs.BaseAudioConfig",
        "TTS.tts.configs.shared_configs.AudioSettings",
        "TTS.tts.configs.shared_configs.SSLModelConfig",
        "TTS.config.shared_configs.BaseDatasetConfig",
        "TTS.tts.models.xtts.AlignerConfig",
        "TTS.tts.models.xtts.VocoderConfig",
        "TTS.tts.models.xtts.XttsArgs",
    ]

    allow = []
    for path in candidates:
        obj = _import_from_string(path)
        if obj is not None:
            allow.append(obj)

    if allow:
        try:
            add_safe_globals(allow)
        except Exception:
            # If add_safe_globals is unavailable or fails, we simply continue.
            pass

class TTSEngine:
    def __init__(self,
                 model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
                 device: Optional[str] = None,
                 output_dir: str = "out_audio",
                 model_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 vocab_path: Optional[str] = None,       # NEW
                 speaker_path: Optional[str] = None,     # NEW (speakers_xtts.pth, optional)
                 use_xtts_inference: bool = True):       # NEW
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        _register_safe_globals()

        self.xtts_model = None
        self.model_dir = model_path  # remember for default ref-wav discovery

        if model_path and config_path and use_xtts_inference:
            # Load *explicit* XTTS model like your headless pipeline
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts

            cfg = XttsConfig()
            cfg.load_json(config_path)
            model = Xtts.init_from_config(cfg)

            # --- resolve checkpoint dir/file robustly ---
            def _resolve_ckpt(model_path: str):
                if model_path and model_path.lower().endswith(".pth"):
                    ckpt_file = model_path
                    ckpt_dir = os.path.dirname(model_path)
                else:
                    ckpt_dir = model_path
                    # prefer common names
                    candidates = ["best_model.pth", "model.pth", "checkpoint.pth"]
                    found = None
                    if ckpt_dir and os.path.isdir(ckpt_dir):
                        for name in candidates:
                            p = os.path.join(ckpt_dir, name)
                            if os.path.exists(p):
                                found = p
                                break
                        # last resort: first .pth in the dir
                        if not found:
                            for f in os.listdir(ckpt_dir):
                                if f.lower().endswith(".pth"):
                                    found = os.path.join(ckpt_dir, f)
                                    break
                    ckpt_file = found
                return ckpt_dir, ckpt_file

            ckpt_dir, ckpt_file = _resolve_ckpt(model_path)

            # vocab / speakers defaults
            vp = vocab_path or os.path.join(os.path.dirname(config_path), "vocab.json")
            sp = speaker_path if (speaker_path and os.path.exists(speaker_path)) else (
                os.path.join(ckpt_dir, "speakers_xtts.pth") if ckpt_dir else None
            )

            # --- build args to match your XTTS version's signature ---
            sig = inspect.signature(model.load_checkpoint)
            kwargs = {
                "vocab_path": vp,
                "speaker_file_path": sp,
                "use_deepspeed": False,
            }

            # choose the correct name for the config parameter
            if "config" in sig.parameters:
                kwargs["config"] = cfg
            elif "cfg" in sig.parameters:
                kwargs["cfg"] = cfg
            else:
                # fallback: pass cfg positionally below if needed
                pass

            # call with checkpoint_dir or checkpoint_path, depending on what's supported
            if "checkpoint_dir" in sig.parameters and ckpt_dir:
                if "config" not in kwargs and "cfg" not in kwargs:
                    # positional fallback: (config, checkpoint_dir=...)
                    model.load_checkpoint(cfg, checkpoint_dir=ckpt_dir, **kwargs)
                else:
                    model.load_checkpoint(checkpoint_dir=ckpt_dir, **kwargs)
            elif "checkpoint_path" in sig.parameters and ckpt_file:
                if "config" not in kwargs and "cfg" not in kwargs:
                    model.load_checkpoint(cfg, checkpoint_path=ckpt_file, **kwargs)
                else:
                    model.load_checkpoint(checkpoint_path=ckpt_file, **kwargs)
            else:
                raise RuntimeError(
                    "XTTS load_checkpoint signature not recognized and/or no checkpoint found. "
                    f"Dir: {ckpt_dir!r}, File: {ckpt_file!r}, Sig: {sig}"
            )

            if torch.cuda.is_available():
                model = model.to("cuda")
            self.xtts_model = model
            self.tts = None
        else:
            # Fallback: Coqui TTS convenience API (old behavior)
            from TTS.api import TTS
            self.tts = TTS(model_path=model_path, config_path=config_path).to(self.device) if (model_path and config_path) else TTS(model_name).to(self.device)

    # ---------- helper: find a default reference wav in the model folder ----------
    def _discover_default_ref_wav(self) -> Optional[str]:
        if not self.model_dir or not os.path.isdir(self.model_dir):
            return None
        # Prefer *_24000.wav, then plain .wav in the folder
        wavs = [f for f in os.listdir(self.model_dir) if f.lower().endswith(".wav")]
        if not wavs:
            return None
        # 24k first
        for f in wavs:
            if f.endswith("_24000.wav"):
                return os.path.join(self.model_dir, f)
        # else: try a file that matches the folder name, e.g., "<basename>.wav"
        base = os.path.basename(os.path.normpath(self.model_dir))
        candidate = os.path.join(self.model_dir, f"{base}.wav")
        if os.path.exists(candidate):
            return candidate
        # otherwise any .wav
        return os.path.join(self.model_dir, wavs[0])

    # ---------- new: XTTS inference path (no need to pass speaker_wav every call) ----------
    def synthesize_infer(self,
                         text: str,
                         language: str = "en",
                         speaker_wav: Optional[str] = None,
                         temperature: float = 0.75,
                         length_penalty: float = 1.0,
                         repetition_penalty: float = 5.0,
                         top_k: int = 50,
                         top_p: float = 0.85,
                         sentence_split: bool = True) -> str:
        """
        Generate audio using XTTS.inference(), computing conditioning latents like run_tts_headless.
        If speaker_wav is not provided, we try to auto-pick one from the model folder (e.g., *_24000.wav).
        """
        if self.xtts_model is None:
            raise RuntimeError("XTTS model not loaded in inference mode. Initialize with use_xtts_inference=True and pass config/checkpoint/vocab.")

        ref = speaker_wav or self._discover_default_ref_wav()
        if not ref or not os.path.exists(ref):
            raise FileNotFoundError("No reference speaker WAV found. Pass speaker_wav or place a reference WAV in the model directory.")

        # Prefer a 24k reference if available (XTTS native rate). If user passed a non-24k wav, model will resample anyway.
        stem, ext = os.path.splitext(ref)
        ref_24k = stem + "_24000.wav"
        if os.path.exists(ref_24k):
            ref = ref_24k

        # Get conditioning latents (mirrors your headless run_tts_headless)
        # Uses config defaults if fields are absent on certain XTTS versions.
        cfg = getattr(self.xtts_model, "config", None)
        gpt_cond_len = getattr(cfg, "gpt_cond_len", 30)
        max_ref_len = getattr(cfg, "max_ref_len", 60)
        sound_norm_refs = getattr(cfg, "sound_norm_refs", False)

        if hasattr(self.xtts_model, "get_conditioning_latents"):
            gpt_cond_latent, speaker_embedding = self.xtts_model.get_conditioning_latents(
                audio_path=ref,
                gpt_cond_len=gpt_cond_len,
                max_ref_length=max_ref_len,
                sound_norm_refs=sound_norm_refs,
            )
        else:
            # Fallback for alt API names
            raise NotImplementedError("Loaded XTTS model lacks get_conditioning_latents(). Update TTS/XTTS version or add an alternate latent extractor.")

        # Run inference like your headless script
        out = self.xtts_model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=float(repetition_penalty),
            top_k=top_k,
            top_p=top_p,
            enable_text_splitting=sentence_split,
        )

        # Save to wav (24 kHz)
        out_path = os.path.join(self.output_dir, f"{uuid.uuid4().hex}.wav")
        if isinstance(out.get("wav"), torch.Tensor):
            wav = out["wav"].detach().cpu()
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
        else:
            wav = torch.tensor(np.asarray(out.get("wav"))).float().unsqueeze(0)

        torchaudio.save(out_path, wav, 24000)
        return out_path

    # ---------- keep your legacy TTS.api path ----------
    def synthesize_to_file(self, text: str,
                           speaker_wavs: Optional[Union[str, List[str]]] = None,
                           language: str = "en") -> str:
        if isinstance(speaker_wavs, str):
            speaker_wavs = [speaker_wavs]
        out_path = os.path.join(self.output_dir, f"{uuid.uuid4().hex}.wav")
        if self.tts is None:
            # If we're in XTTS inference mode, delegate to synthesize_infer
            return self.synthesize_infer(text=text, language=language, speaker_wav=(speaker_wavs[0] if speaker_wavs else None))
        self.tts.tts_to_file(text=text, file_path=out_path, speaker_wav=speaker_wavs, language=language)
        return out_path