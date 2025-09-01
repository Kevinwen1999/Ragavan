# tts_engine.py
"""
Minimal TTS wrapper for Discord bot (Coqui XTTS v2) with PyTorch 2.6+
safe-globals allow-list for XTTS checkpoints.

- Zero/low-shot voice cloning via reference WAVs (speaker_wav list)
- CPU/GPU supported; GPU recommended
"""

import os
import uuid
from typing import List, Optional, Union
import importlib

import torch


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
                 config_path: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        # --- allow-list block you already added for PyTorch 2.6 stays here ---
        _register_safe_globals()

        from TTS.api import TTS

        if model_path and config_path:
            # Load your fine-tuned checkpoint
            self.tts = TTS(model_path=model_path, config_path=config_path).to(self.device)
        else:
            # Fallback to base model by name
            self.tts = TTS(model_name).to(self.device)

    def synthesize_to_file(self, text: str,
                           speaker_wavs: Optional[Union[str, List[str]]] = None,
                           language: str = "en") -> str:
        if isinstance(speaker_wavs, str):
            speaker_wavs = [speaker_wavs]
        out_path = os.path.join(self.output_dir, f"{uuid.uuid4().hex}.wav")
        self.tts.tts_to_file(text=text, file_path=out_path,
                             speaker_wav=speaker_wavs, language=language)
        return out_path