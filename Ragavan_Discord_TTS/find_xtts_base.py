# find_xtts_base.py
from TTS.api import TTS

if __name__ == "__main__":
    t = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    syn = t.synthesizer
    # These attributes exist on current TTS; prints useful paths
    print("tts_checkpoint:", syn.tts_checkpoint)
    print("tts_config_path:", syn.tts_config_path)
    if getattr(syn, "vocoder_checkpoint", None):
        print("vocoder_checkpoint:", syn.vocoder_checkpoint)
        print("vocoder_config_path:", syn.vocoder_config_path)