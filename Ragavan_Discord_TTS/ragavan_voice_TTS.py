# bot.py
import asyncio
import json
import os
import textwrap
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import discord
from discord.ext import commands
from dotenv import load_dotenv

from tts_engine import TTSEngine

# -------------------
# Configuration
# -------------------
# 1) Put your bot token in the environment:
#    set DISCORD_BOT_TOKEN=xxxx (Windows)
#    export DISCORD_BOT_TOKEN=xxxx (macOS/Linux)
# Load the environment variables from .env file
load_dotenv()

BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not BOT_TOKEN:
    raise SystemExit("Please set DISCORD_BOT_TOKEN environment variable.")

# 2) ffmpeg must be installed and on PATH.
FFMPEG_EXE = "ffmpeg"

# 3) Default language for TTS
DEFAULT_LANG = "en"

# 4) Voices directory (JSON profiles made by make_voice_profile.py)
VOICES_DIR = Path("voices")

# Supported language shortcuts -> model codes
LANG_ALIASES = {
    "en": "en",
    "english": "en",
    "zh": "zh-cn",
    "cn": "zh-cn",
    "zh-cn": "zh-cn",   # Mandarin (Simplified)
    "zh_cn": "zh-cn",
    "zh-tw": "zh-tw",   # Mandarin (Traditional)
    "zh_tw": "zh-tw",
    "jp": "ja",
    "kr": "ko"
    # add more if you like, e.g. "jp":"ja", "kr":"ko", etc.
}

# -------------------
# Helpers
# -------------------
def discover_voice_profiles() -> Dict[str, List[str]]:
    profiles = {}
    if VOICES_DIR.exists():
        for p in VOICES_DIR.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                samples = payload.get("samples") or []
                name = payload.get("name") or p.stem
                if samples:
                    profiles[name.lower()] = samples
            except Exception:
                traceback.print_exc()
    return profiles


def chunk_text(text: str, max_len: int = 400) -> List[str]:
    # Avoid extremely long single calls. XTTS can handle long text, but chunking improves stability.
    lines = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        while len(para) > max_len:
            # split on nearest space
            cut = para.rfind(" ", 0, max_len)
            if cut == -1:
                cut = max_len
            lines.append(para[:cut])
            para = para[cut:].strip()
        if para:
            lines.append(para)
    return lines if lines else ["..."]


@dataclass
class GuildState:
    current_voice: Optional[str] = None  # key in profiles dict
    speaking_lock: asyncio.Lock = asyncio.Lock()
    current_lang: str = DEFAULT_LANG


# -------------------
# Bot Setup
# -------------------
intents = discord.Intents.all()
intents.message_content = True  # enable in Dev Portal for your bot
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

# Shared TTS engine + voices
# tts_engine = TTSEngine()
tts_engine = TTSEngine(
    model_path=r"runs\xtts_ft_user\testing",
    config_path=r"runs\xtts_ft_user\testing\config.json",
)
voice_profiles = discover_voice_profiles()
guild_states: Dict[int, GuildState] = {}


def get_guild_state(guild_id: int) -> GuildState:
    if guild_id not in guild_states:
        guild_states[guild_id] = GuildState()
    return guild_states[guild_id]


async def ensure_voice(ctx: commands.Context) -> Optional[discord.VoiceClient]:
    """
    Ensure the bot is connected to the author's voice channel.
    """
    if not ctx.author or not isinstance(ctx.author, discord.Member):
        await ctx.reply("I can't find your voice channel.")
        return None

    if ctx.voice_client and ctx.voice_client.is_connected():
        return ctx.voice_client

    if ctx.author.voice and ctx.author.voice.channel:
        try:
            return await ctx.author.voice.channel.connect()
        except Exception as e:
            await ctx.reply(f"Failed to join: {e}")
            return None
    else:
        await ctx.reply("Join a voice channel first, then use `!join` or `!say`.")
        return None


# -------------------
# Commands
# -------------------
@bot.command(name="help")
async def _help(ctx: commands.Context):
    msg = textwrap.dedent(
        """
        **Commands**
        `!join` — Join your current voice channel
        `!leave` — Leave voice
        `!voices` — List available voice profiles
        `!setvoice <name>` — Use a voice profile (from `voices/*.json`)
        `!say <text>` — Speak the text with current voice profile (or default)

        **Examples**
        `!voices`
        `!setvoice myvoice`
        `!say Hello everyone, this is a custom voice.`
        `!lang <code>` — Set language (e.g., `!lang zh`, `!lang zh-cn`, `!lang en`)
        """
    )
    await ctx.reply(msg)


@bot.command(name="join")
async def join(ctx: commands.Context):
    vc = await ensure_voice(ctx)
    if vc:
        await ctx.reply(f"Joined **{vc.channel}**.")


@bot.command(name="leave")
async def leave(ctx: commands.Context):
    if ctx.voice_client:
        await ctx.voice_client.disconnect(force=True)
        await ctx.reply("Left voice channel.")
    else:
        await ctx.reply("I'm not connected to a voice channel.")


@bot.command(name="lang")
async def set_language(ctx: commands.Context, code: Optional[str] = None):
    if not code:
        st = get_guild_state(ctx.guild.id)
        return await ctx.reply(f"Current language: **{st.current_lang}**. "
                               f"Try `!lang zh` or `!lang zh-cn`.")

    norm = LANG_ALIASES.get(code.lower())
    if not norm:
        supported = ", ".join(sorted(set(LANG_ALIASES.keys())))
        return await ctx.reply(f"Unsupported code `{code}`. Try one of: {supported}")

    st = get_guild_state(ctx.guild.id)
    st.current_lang = norm
    await ctx.reply(f"Language set to **{norm}**.")

@bot.command(name="voices")
async def voices(ctx: commands.Context):
    global voice_profiles
    voice_profiles = discover_voice_profiles()
    if not voice_profiles:
        await ctx.reply("No voice profiles found. Create one with `make_voice_profile.py`.")
        return
    listing = "\n".join(f"- {name} ({len(paths)} samples)" for name, paths in voice_profiles.items())
    await ctx.reply(f"**Available voices:**\n{listing}")


@bot.command(name="setvoice")
async def setvoice(ctx: commands.Context, name: Optional[str] = None):
    if not name:
        await ctx.reply("Usage: `!setvoice <name>` (see `!voices`)")
        return
    name_key = name.lower()
    if name_key not in voice_profiles:
        await ctx.reply(f"Voice '{name}' not found. Use `!voices` to see options.")
        return
    st = get_guild_state(ctx.guild.id)
    st.current_voice = name_key
    await ctx.reply(f"Voice set to **{name}**.")


@bot.command(name="say")
async def say(ctx: commands.Context, *, text: str):
    if not text.strip():
        await ctx.reply("Give me some text, e.g., `!say Hello!`")
        return

    vc = await ensure_voice(ctx)
    if not vc:
        return

    st = get_guild_state(ctx.guild.id)
    # capture the profile at call time
    speaker_wavs = voice_profiles.get(st.current_voice) if st.current_voice else None

    # serialize speaking to avoid overlapping audio
    async with st.speaking_lock:
        try:
            parts = chunk_text(text, max_len=450)
            for i, part in enumerate(parts, 1):
                wav_path = tts_engine.synthesize_to_file(
                    text=part,
                    speaker_wavs=speaker_wavs,
                    language=get_guild_state(ctx.guild.id).current_lang,  # <— changed
                )
                source = discord.FFmpegPCMAudio(
                    wav_path,
                    executable=FFMPEG_EXE,
                    before_options="-nostdin",
                    options="-vn -f s16le -ar 48000 -ac 2",
                )
                vc.play(source)
                # block until finished
                while vc.is_playing():
                    await asyncio.sleep(0.2)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(f"TTS failed: `{e}`")
        finally:
            # Optional: clean up generated files to save disk
            pass


# -------------------
# Run
# -------------------
if __name__ == "__main__":
    print("Starting bot...")
    bot.run(BOT_TOKEN)
