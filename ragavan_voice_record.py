import os
from datetime import datetime
from pathlib import Path

import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Where to save WAVs (absolute, next to this script)
BASE_DIR = Path(__file__).resolve().parent / "recordings_wav"
BASE_DIR.mkdir(parents=True, exist_ok=True)

bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

# Cache voice clients per guild (as the guide recommends) :contentReference[oaicite:1]{index=1}
connections = {}


def ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


async def once_done(sink: discord.sinks.Sink, channel: discord.TextChannel, *args):
    """
    Called automatically when vc.stop_recording() is invoked. :contentReference[oaicite:2]{index=2}
    Saves separate WAV per user locally.
    """
    guild_id = channel.guild.id if channel.guild else 0
    out_dir = BASE_DIR / str(guild_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for user_id, audio in sink.audio_data.items():  # per-user audio dict :contentReference[oaicite:3]{index=3}
        # audio.file is a file-like object (BytesIO or temp file depending on version)
        out_path = out_dir / f"{user_id}_{ts()}.{sink.encoding}"
        audio.file.seek(0)
        with open(out_path, "wb") as f:
            f.write(audio.file.read())
        saved.append(str(out_path))

        # Important: close/cleanup per-user audio object if available
        try:
            audio.cleanup()
        except Exception:
            pass

    # Disconnect after finishing (matches guide behavior) :contentReference[oaicite:4]{index=4}
    try:
        await sink.vc.disconnect()
    except Exception:
        pass

    if saved:
        await channel.send(f"Saved {len(saved)} WAV file(s) to: `{out_dir}`")
    else:
        await channel.send("Stopped recording, but no user audio was captured (nobody spoke?).")


@bot.command()
async def record(ctx: commands.Context):
    voice = ctx.author.voice
    if not voice or not voice.channel:
        return await ctx.send("You aren't in a voice channel.")

    # Connect and start recording
    vc = await voice.channel.connect()
    connections[ctx.guild.id] = vc

    vc.start_recording(
        discord.sinks.WaveSink(),  # records WAV per user in sink.audio_data :contentReference[oaicite:5]{index=5}
        once_done,
        ctx.channel,
    )
    await ctx.send("Started recording. Use `!stop` to stop and save per-user WAVs.")


@bot.command()
async def stop(ctx: commands.Context):
    vc = connections.get(ctx.guild.id)
    if not vc:
        return await ctx.send("I am not recording in this server.")

    vc.stop_recording()  # triggers once_done callback :contentReference[oaicite:6]{index=6}
    del connections[ctx.guild.id]
    await ctx.send("Stopping recording...")


@bot.command()
async def leave(ctx: commands.Context):
    vc = ctx.voice_client
    if vc:
        try:
            if getattr(vc, "recording", False):
                vc.stop_recording()
        except Exception:
            pass
        await vc.disconnect()
    await ctx.send("Disconnected.")


bot.run(TOKEN)