# train_xtts.py
import os, sys, json, gc, importlib, warnings
from pathlib import Path
from typing import Optional, Tuple, List

# Silence the pkg_resources deprecation warning from librosa
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

# CUDA DLLs early (Windows)
if sys.platform == "win32":
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
    if os.path.isdir(cuda_bin):
        try:
            os.add_dll_directory(cuda_bin)
            print(f"[dll] added: {cuda_bin}")
        except Exception:
            pass

def _register_safe_globals():
    try:
        from torch.serialization import add_safe_globals
    except Exception:
        return
    candidates = [
        # Common XTTS objects seen in checkpoints
        "TTS.tts.configs.xtts_config.XttsConfig",
        "TTS.tts.models.xtts.XttsAudioConfig",
        # Extra likely configs/types
        "TTS.tts.configs.shared_configs.BaseAudioConfig",
        "TTS.tts.configs.shared_configs.AudioSettings",
        "TTS.tts.configs.shared_configs.SSLModelConfig",
        "TTS.config.shared_configs.BaseDatasetConfig",
        "TTS.tts.models.xtts.AlignerConfig",
        "TTS.tts.models.xtts.VocoderConfig",
        "TTS.tts.models.xtts.XttsArgs",
    ]
    allow = []
    for dotted in candidates:
        try:
            mod, name = dotted.rsplit(".", 1)
            obj = getattr(importlib.import_module(mod), name, None)
            if obj is not None:
                allow.append(obj)
        except Exception:
            pass
    if allow:
        try:
            add_safe_globals(allow)
            print(f"[safe-globals] registered {len(allow)} XTTS classes")
        except Exception:
            pass

from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import (
    GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
)
from TTS.utils.manage import ModelManager

def _load_json_config():
    cfg_path = Path(os.environ.get("FT_CONFIG_PATH", "finetune_xtts.json"))
    if not cfg_path.exists():
        raise SystemExit(f"Config file not found: {cfg_path.resolve()}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[config] Loaded: {cfg_path}")
    return data

def _count_lines(p: Path) -> int:
    if not p.exists(): return 0
    with p.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f if _.strip())

def _find_trainer_checkpoints(root: Path) -> List[Path]:
    """Return parent dirs that contain a Trainer 'checkpoint.pth' file, newest first."""
    if not root.exists():
        return []
    hits = []
    for p in root.rglob("checkpoint.pth"):
        try:
            hits.append(p.parent)
        except Exception:
            pass
    hits.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return hits

def _classify_resume_target(p: Path) -> Tuple[Optional[str], Optional[Path]]:
    """
    Heuristics:
      - if directory containing 'checkpoint.pth' => ('trainer', dir)
      - if file named 'checkpoint.pth'          => ('trainer', dir_of_file)
      - if file .pth (e.g., export/model.pth)   => ('xtts', file)
      - else                                    => (None, None)
    """
    if p.is_dir() and (p / "checkpoint.pth").exists():
        return "trainer", p
    if p.is_file() and p.name == "checkpoint.pth":
        return "trainer", p.parent
    if p.is_file() and p.suffix == ".pth":
        return "xtts", p
    return None, None

def main():
    cfg_json = _load_json_config()

    # ---- JSON -> local vars
    run_name           = cfg_json.get("run_name", "xtts_ft_user")
    out_dir            = Path(cfg_json.get("output_path", "runs/xtts_ft_user")).resolve()
    epochs             = int(cfg_json.get("epochs", 15))
    batch_size         = int(cfg_json.get("batch_size", 1))
    num_workers        = int(cfg_json.get("num_loader_workers", 2))
    mixed_precision    = bool(cfg_json.get("mixed_precision", True))
    print_step         = int(cfg_json.get("print_step", 25))
    checkpoint_every   = int(cfg_json.get("checkpoint_every", 1000))
    save_n_checkpoints = int(cfg_json.get("save_n_checkpoints", 5))
    lr                 = float(cfg_json.get("lr", 5e-6))
    optimizer_name     = str(cfg_json.get("optimizer", "adamw")).lower()
    lr_sched_name      = str(cfg_json.get("lr_scheduler", "cosine")).lower()
    grad_clip          = float(cfg_json.get("grad_clip", 1.0))
    seed               = int(cfg_json.get("seed", 42))
    eval_split_max_size = int(cfg_json.get("eval_split_max_size", 64))
    grad_accum         = int(cfg_json.get("grad_accum", cfg_json.get("grad_accum_steps", 1)))
    run_eval           = bool(cfg_json.get("run_eval", True))

    # NEW: resume knobs
    resume_from         = cfg_json.get("resume_from")  # str or None
    resume_from_latest  = bool(cfg_json.get("resume_from_latest", False))
    xtts_ckpt_override  = cfg_json.get("xtts_checkpoint_override")  # str or None

    ds_list = cfg_json.get("datasets", [])
    if not ds_list:
        raise SystemExit("No datasets[] specified in config.")
    ds0 = ds_list[0]
    ds_path    = Path(ds0.get("path", "data/processed_auto"))
    meta_train = ds0.get("meta_file_train", "metadata.csv")
    meta_val   = ds0.get("meta_file_val", "metadata_val.csv")
    language   = ds0.get("language", cfg_json.get("language", "en"))

    out_dir.mkdir(parents=True, exist_ok=True)
    if not (ds_path / meta_train).exists():
        raise SystemExit(f"Missing {meta_train} in {ds_path}")
    if not (ds_path / meta_val).exists():
        # if no val file, disable eval automatically
        run_eval = False

    # Preflight: warn if val is empty (avoids crash)
    train_lines = _count_lines(ds_path / meta_train)
    val_lines   = _count_lines(ds_path / meta_val) if run_eval else 0
    if run_eval and val_lines == 0:
        print("[warn] metadata_val.csv is empty after preflight. Disabling eval this run.")
        run_eval = False

    # ---- Base XTTS assets
    base_dir = out_dir / "XTTS_v2_base"
    base_dir.mkdir(parents=True, exist_ok=True)
    DVAE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    MEL_LINK  = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
    TOK_LINK  = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    CFG_LINK  = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json"
    CKPT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

    dvae_path = base_dir / os.path.basename(DVAE_LINK)
    mel_path  = base_dir / os.path.basename(MEL_LINK)
    tok_path  = base_dir / os.path.basename(TOK_LINK)
    cfg_path  = base_dir / os.path.basename(CFG_LINK)
    ckpt_path = base_dir / os.path.basename(CKPT_LINK)

    need = [p for p in (dvae_path, mel_path, tok_path, cfg_path, ckpt_path) if not p.exists()]
    if need:
        print("[download] Fetching base XTTS v2 files...")
        ModelManager._download_model_files(
            [DVAE_LINK, MEL_LINK, TOK_LINK, CFG_LINK, CKPT_LINK],
            str(base_dir),
            progress_bar=True
        )

    # Allow-list checkpoint classes (PyTorch 2.6+)
    _register_safe_globals()

    # ---- Dataset config
    dataset_cfg = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name="ft_dataset",
        path=str(ds_path),
        meta_file_train=meta_train,
        meta_file_val=meta_val,
        language=language,
    )

    # ---- Trainer config (seeded from base config.json)
    MAX_AUDIO = int(cfg_json.get("max_audio_len", 255995))
    MAX_TEXT  = int(cfg_json.get("max_text_len", 200))

    # If user overrides XTTS weights for warm-start, honor it
    if xtts_ckpt_override:
        override_p = Path(xtts_ckpt_override)
        if override_p.exists():
            ckpt_path = override_p
            print(f"[resume] Using xtts_checkpoint_override: {ckpt_path}")
        else:
            print(f"[resume] xtts_checkpoint_override not found: {override_p} (ignored)")

    model_args = GPTArgs(
        max_conditioning_length=264600,
        min_conditioning_length=88200,
        max_wav_length=MAX_AUDIO,
        max_text_length=MAX_TEXT,
        mel_norm_file=str(mel_path),
        dvae_checkpoint=str(dvae_path),
        xtts_checkpoint=str(ckpt_path),  # may be overridden above
        tokenizer_file=str(tok_path),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_cfg = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    tcfg = GPTTrainerConfig()
    tcfg.load_json(str(cfg_path))
    tcfg.model_args = model_args
    tcfg.audio = audio_cfg

    tcfg.output_path = str(out_dir)
    tcfg.run_name = f"GPT_{run_name}"
    tcfg.project_name = "XTTS_trainer"
    tcfg.dashboard_logger = "tensorboard"

    tcfg.epochs = epochs
    tcfg.batch_size = batch_size
    tcfg.num_loader_workers = num_workers
    tcfg.mixed_precision = mixed_precision
    tcfg.print_step = print_step
    tcfg.plot_step = max(100, print_step)
    tcfg.log_model_step = max(100, print_step)
    tcfg.save_step = checkpoint_every
    tcfg.save_n_checkpoints = save_n_checkpoints
    tcfg.save_checkpoints = True
    tcfg.grad_clip = grad_clip
    tcfg.seed = seed
    tcfg.grad_accum_steps = grad_accum
    tcfg.run_eval = run_eval  # honor JSON / preflight

    if optimizer_name == "adamw":
        tcfg.optimizer = "AdamW"
        tcfg.optimizer_wd_only_on_weights = True
        tcfg.optimizer_params = {"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2}
    else:
        tcfg.optimizer = "Adam"
        tcfg.optimizer_params = {"betas": [0.9, 0.98], "eps": 1e-8}

    tcfg.lr = lr
    if "cos" in lr_sched_name:
        tcfg.lr_scheduler = "CosineAnnealingLR"
        tcfg.lr_scheduler_params = {"T_max": max(checkpoint_every * 10, 1000), "eta_min": lr * 0.1}
    else:
        tcfg.lr_scheduler = "MultiStepLR"
        tcfg.lr_scheduler_params = {
            "milestones": [checkpoint_every*3, checkpoint_every*6, checkpoint_every*9],
            "gamma": 0.5
        }

    # ---- Init model (with current config / base or override weights)
    model = GPTTrainer.init_from_config(tcfg)

    # ---- Decide resume target
    trainer_restore_path: Optional[str] = None

    if resume_from:
        p = Path(resume_from)
        kind, target = _classify_resume_target(p)
        if kind == "trainer":
            trainer_restore_path = str(target)
            print(f"[resume] Trainer checkpoint restore: {trainer_restore_path}")
        elif kind == "xtts":
            # Replace current XTTS weights with provided file (warm start)
            try:
                import torch
                state = torch.load(str(target), map_location="cpu", weights_only=False)
                sd = state.get("model", state)
                if hasattr(model, "xtts") and hasattr(model.xtts, "load_state_dict"):
                    model.xtts.load_state_dict(sd, strict=False)
                    print(f"[resume] Loaded XTTS weights from {target} (strict=False)")
                else:
                    print("[resume] Warning: model.xtts not found; skipping xtts resume.")
            except Exception as e:
                print(f"[resume] Failed to load XTTS .pth: {target} ({e})")
        else:
            print(f"[resume] Could not classify resume_from: {p} (ignored)")

    elif resume_from_latest:
        # Search under out_dir for latest Trainer checkpoint
        ckpt_dirs = _find_trainer_checkpoints(out_dir)
        if ckpt_dirs:
            trainer_restore_path = str(ckpt_dirs[0])
            print(f"[resume] Auto-picked latest Trainer checkpoint: {trainer_restore_path}")
        else:
            print("[resume] resume_from_latest requested but no checkpoints found under output_path.")

    # ---- Load dataset splits
    train_samples, eval_samples = load_tts_samples(
        [dataset_cfg],
        eval_split=tcfg.run_eval,
        eval_split_max_size=eval_split_max_size,
        eval_split_size=tcfg.eval_split_size,
    )

    # ---- Trainer
    trainer = Trainer(
        TrainerArgs(
            restore_path=trainer_restore_path,  # <-- resume optimizer/scaler/steps if provided
            start_with_eval=False,
            grad_accum_steps=grad_accum,
        ),
        tcfg,
        output_path=str(Path(out_dir) / "run" / "training"),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples if tcfg.run_eval else None,
    )

    # Fit
    trainer.fit()

    # --- end-of-run export (always write a final model) ---
    try:
        import torch, shutil
        export_dir = Path(out_dir) / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        export_ckpt = export_dir / "model.pth"

        xtts_model = getattr(model, "xtts", None)
        if xtts_model is not None:
            torch.save({"model": xtts_model.state_dict()}, export_ckpt)
            shutil.copy2(str(base_dir / "config.json"), export_dir / "config.json")
            shutil.copy2(str(base_dir / "vocab.json"), export_dir / "vocab.json")
            print(f"[export] Wrote {export_ckpt}")
        else:
            print("[export] Could not find model.xtts; skipped export.")
    except Exception as e:
        print("[export] Failed:", e)

    del model, trainer, train_samples, eval_samples
    gc.collect()
    print("[done] Training complete.")

if __name__ == "__main__":
    main()
