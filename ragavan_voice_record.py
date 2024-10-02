from datetime import datetime
import os
import wave
import discord
from discord.ext import commands, voice_recv
import asyncio
import numpy as np
from discord.ext import commands
from dotenv import load_dotenv


# Load the environment variables from .env file
load_dotenv()

# Get the Discord bot token from .env
TOKEN = os.getenv('DISCORD_BOT_TOKEN')

# Directory to store user audio files
BASE_DIR = "recordings"


bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

# Dictionary to keep track of user audio files
audio_files = {}

class MySink(voice_recv.AudioSink):
    def __init__(self):
        super().__init__()

    def wants_opus(self) -> bool:
        """Return False to receive PCM (uncompressed) audio, True for Opus."""
        return False  # Set to True if you want Opus instead of PCM

    def write(self, user, data):
        """Process the incoming voice data and save it to a file."""
        user_id = user.id if user else "unknown_user"
        
        # Ensure a file is created for each user
        if user_id not in audio_files:
            self.create_audio_file(user_id)
        
        # Append the PCM data to the user's audio file
        audio_files[user_id].writeframes(data.pcm)
    
    def create_audio_file(self, user_id):
        """Create a new WAV file in a unique directory for a user to store PCM data."""
        # Create a directory for the user if it doesn't exist
        user_dir = os.path.join(BASE_DIR, str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        
        # Create a unique filename with a timestamp to avoid overwriting
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{user_id}_{timestamp}.wav"
        filepath = os.path.join(user_dir, filename)
        
        wf = wave.open(filepath, 'wb')
        wf.setnchannels(2)  # Stereo channel
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(48000)  # 48kHz sample rate
        audio_files[user_id] = wf
        print(f"Created audio file for user {user_id}: {filepath}")

    def cleanup(self):
        """Cleanup after recording is done. Close all open files."""
        for user_id, wf in audio_files.items():
            wf.close()
            print(f"Closed audio file for user {user_id}")
        audio_files.clear()

@bot.command()
async def join(ctx):
    """Join a voice channel and start recording."""
    if ctx.author.voice:
        vc = await ctx.author.voice.channel.connect(cls=voice_recv.VoiceRecvClient)
        vc.listen(MySink())

@bot.command()
async def leave(ctx):
    """Leave the voice channel."""
    if ctx.voice_client:
        ctx.voice_client.stop_listening() 
        await ctx.voice_client.disconnect()

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name}")

# Run the bot
bot.run(TOKEN)