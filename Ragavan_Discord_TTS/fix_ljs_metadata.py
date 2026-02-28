# fix_ljs_metadata.py
from pathlib import Path
import shutil

def fix_file(p: Path):
    if not p.exists(): 
        return
    backup = p.with_suffix(p.suffix + ".bak")
    shutil.copyfile(p, backup)
    out_lines = []
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln.strip():
                continue
            parts = ln.split("|")
            if len(parts) == 1:
                # If line is just "<utt_id>", skip
                continue
            elif len(parts) == 2:
                utt, text = parts
                # duplicate text as normalized_text (you can plug a real normalizer later)
                out_lines.append(f"{utt}|{text}|{text}\n")
            else:
                # already >= 3 columns -> keep as-is
                out_lines.append(ln + "\n")
    with p.open("w", encoding="utf-8", newline="") as f:
        f.writelines(out_lines)
    print(f"Fixed {p} (backup at {backup})")

for name in ("data/processed/metadata.csv",
             "data/processed/metadata_val.csv",
             "data/processed_auto/metadata.csv",
             "data/processed_auto/metadata_val.csv"):
    fix_file(Path(name))
