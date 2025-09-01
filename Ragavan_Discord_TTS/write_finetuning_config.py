# write_finetune_config.py
import json
from pathlib import Path

CFG = {
  "model": "xtts",
  "run_name": "xtts_ft_user",
  "output_path": "runs/xtts_ft_user",
  "batch_size": 8,
  "eval_batch_size": 8,
  "num_loader_workers": 2,
  "eval_split_max_size": 64,
  "run_eval": False,
  "epochs": 30,
  "optimizer": "adamw",
  "lr": 5e-5,
  "grad_clip": 1.0,
  "lr_scheduler": "cosine",
  "mixed_precision": False,
  "print_step": 25,
  "checkpoint_every": 99999999,
  "save_n_checkpoints": 1,
  "seed": 42,
  "grad_accum": 4,
  "max_audio_len": 330750,
  "max_text_len": 300,
  "language": "zh",

  "datasets": [
    {
      "formatter": "ljspeech",
      "path": "data/processed_auto",
      "meta_file_train": "metadata.csv",
      "meta_file_val": "metadata_val.csv"
    }
  ],

  # You can keep language fixed during FT (prosody/timbre adaptation).
  # For JA/ZH datasets, the tokenizer will still see those characters.
  "use_phonemes": False
}

if __name__ == "__main__":
    out = Path("finetune_xtts.json")
    out.write_text(json.dumps(CFG, indent=2), encoding="utf-8")
    print("Wrote", out)