import io
import queue
import discord
import asyncio
import edge_tts
import pyaudio
import numpy as np
from discord.ext import commands
import scipy
from scipy.signal import resample
from dotenv import load_dotenv
import os
import subprocess
import yt_dlp

# Experimental file for real time voice data processing, not very successful yet

# Load the environment variables from .env file
load_dotenv()

# Get the Discord bot token from .env
TOKEN = os.getenv('DISCORD_BOT_TOKEN')

intents = discord.Intents.all()
bot = commands.Bot(command_prefix='!', intents=intents)

# Constants for Audio Processing
CHUNK = 8192  # Number of audio samples per frame
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Single channel for mic input (mono)
RATE = 44100  # Sample rate (samples per second)

# Set up PyAudio for mic input
audio = pyaudio.PyAudio()

audio_buffer = queue.Queue()

# Open microphone stream
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

music_queues = {}  # guild_id -> asyncio.Queue()


@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')

@bot.command()
async def ping(ctx):
    await ctx.send("Pong!")

@bot.command()
async def join(ctx):
    # 1) Must be in a VC
    if not ctx.author.voice:
        return await ctx.send("❌ You need to be in a voice channel first!")

    channel = ctx.author.voice.channel

    # 2) If already connected, disconnect first
    if ctx.voice_client and ctx.voice_client.is_connected():
        await ctx.voice_client.disconnect()

    # 3) Try connecting with timeout + no built-in retry
    try:
        vc = await channel.connect(timeout=20.0, reconnect=False)
    except discord.errors.ConnectionClosed as e:
        # Discord told us our session is invalid (4006) after handshake
        if getattr(e, 'code', None) == 4006:
            # Optional: retry once more
            try:
                vc = await channel.connect(timeout=20.0, reconnect=False)
            except Exception:
                return await ctx.send("⚠️ Voice session expired and reconnect failed. Please try again.")
        else:
            return await ctx.send(f"⚠️ Failed to connect to voice channel: {e}")
    except Exception as e:
        # Some other error (DNS, permissions, etc)
        return await ctx.send(f"⚠️ Could not join voice channel: {e}")

    # 4) Success
    await ctx.send(f"✅ Joined **{vc.channel.name}**!")

@bot.command()
async def leave(ctx):
    if ctx.voice_client:
        await ctx.guild.voice_client.disconnect()

@bot.command()
async def start(ctx):
    if ctx.voice_client and ctx.voice_client.is_connected():
        vc = ctx.voice_client
        bot.loop.create_task(capture_mic(vc))
        bot.loop.create_task(play_buffered_audio(vc))

@bot.command()
async def test_audio(ctx):
    if ctx.voice_client is None:
        if ctx.author.voice:
            channel = ctx.author.voice.channel
            await channel.connect()

    vc = ctx.voice_client

    # Ensure ffmpeg is available
    source = discord.FFmpegPCMAudio('tts_rvc_output.wav')
    vc.play(source)

@bot.command()
async def play_test_audio(ctx):
    if ctx.voice_client is None:
        if ctx.author.voice:
            channel = ctx.author.voice.channel
            await channel.connect()

    vc = ctx.voice_client

    # Read the PCM file into memory
    with open('test_audio.pcm', 'rb') as f:
        pcm_data = f.read()

    # Use BytesIO to create an in-memory stream of the audio data
    audio_stream = io.BytesIO(pcm_data)

    # Play the audio using the in-memory stream
    vc.play(discord.PCMAudio(audio_stream), after=lambda e: audio_stream.close())

    await ctx.send("Playing test audio")


async def text_to_speech(line_text, output_audio_file, voice_template="zh-CN-shaanxi-XiaoniNeural", rate_="+0%"):
    # Create an instance of the Communication class
    # voice_template = "zh-CN-shaanxi-XiaoniNeural"
    tts = edge_tts.Communicate(line_text, voice=voice_template, rate=rate_)

    # Generate temporary audio file for the line
    temp_file = output_audio_file
    await tts.save(temp_file)

    return temp_file

@bot.command()
async def tts(ctx, text, voice_template="zh-CN-XiaoyiNeural", rate="+0%"):
    result = await text_to_speech(text, "tts_output.mp3", voice_template, rate)
    vc = ctx.voice_client
    source = discord.FFmpegPCMAudio('tts_output.mp3')
    vc.play(source)


async def stop_playing(ctx):
    if ctx.voice_client and ctx.voice_client.is_playing():
        ctx.voice_client.stop()  # Stops the current audio
        await ctx.send("Music stopped!")
        
    else:
        await ctx.send("Nothing is playing right now.")
        

async def fetch_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
    }
    # url input from user 
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            print(f"INFO URL IS {info['url']}")
            return info['url']
        print("Audio downloaded successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
@bot.command()
async def playYoutube(ctx, url):
    guild_id = ctx.guild.id
    if guild_id not in music_queues:
        music_queues[guild_id] = asyncio.Queue()

    await music_queues[guild_id].put(url)
    await ctx.send(f"Added to queue: {url}")

    # Start the player if not already playing
    if not ctx.voice_client.is_playing():
        bot.loop.create_task(play_music_queue(ctx))

async def play_music_queue(ctx):
    guild_id = ctx.guild.id
    vc = ctx.voice_client

    while not music_queues[guild_id].empty():
        url = await music_queues[guild_id].get()
        audio_url = await fetch_audio(url)
        source = discord.FFmpegPCMAudio(audio_url)

        def after_playing(error):
            if error:
                print(f"Error in after_playing: {error}")
            # Continue playing next song
            fut = asyncio.run_coroutine_threadsafe(play_music_queue(ctx), bot.loop)
            try:
                fut.result()
            except Exception as e:
                print(f"Error running coroutine: {e}")

        vc.play(source, after=after_playing)
        await ctx.send(f"Now playing: {url}")
        
        break  # Break to wait for current song to end (after() will restart this)

@bot.command()
async def skip(ctx):
    if ctx.voice_client and ctx.voice_client.is_playing():
        ctx.voice_client.stop()
        await ctx.send("Skipped current song!")
    else:
        await ctx.send("No song is currently playing.")

@bot.command()
async def queue(ctx):
    guild_id = ctx.guild.id
    if guild_id not in music_queues or music_queues[guild_id].empty():
        await ctx.send("The music queue is empty!")
    else:
        # List the URLs currently in the queue
        queue_list = list(music_queues[guild_id]._queue)
        message = "**Current Queue:**\n"
        for i, url in enumerate(queue_list, start=1):
            message += f"{i}. {url}\n"
        await ctx.send(message)

@bot.command()
async def clear(ctx):
    guild_id = ctx.guild.id

    if guild_id in music_queues and not music_queues[guild_id].empty():
        music_queues[guild_id] = asyncio.Queue()  # Just create a new empty queue
        await ctx.send("Music queue cleared!")
    else:
        await ctx.send("The queue is already empty!")

@bot.command()
async def stop(ctx):
    await stop_playing(ctx)

def save_processed_audio(raw_data, input_sample_rate=44100, input_channels=1, filename="test_audio.pcm"):

    # Convert the byte data to numpy array (assuming it's 16-bit audio)
    audio_data = np.frombuffer(raw_data, dtype=np.int16)

    # Resample audio to 48000 Hz
    if input_sample_rate != 48000:
        target_sample_rate = 48000
        audio_data = scipy.signal.resample(audio_data, int(len(audio_data) * target_sample_rate / input_sample_rate))

    # If input audio is mono (1 channel), convert to stereo by duplicating channels
    if input_channels == 1:
        audio_data = np.repeat(audio_data, 2)

    # Save the processed audio to a file
    with open(filename, "wb") as f:
        f.write(audio_data.astype(np.int16).tobytes())

def process_audio(raw_data, input_sample_rate=44100, input_channels=1,):
    """Alter pitch and tone here"""
    audio_data = np.frombuffer(raw_data, dtype=np.int16)

    # Example pitch shift (speeding up the audio for higher pitch)
    audio_data = resample(audio_data, int(len(audio_data) * 1.5))  # Increase pitch


    # Convert the byte data to numpy array (assuming it's 16-bit audio)
    audio_data = np.frombuffer(raw_data, dtype=np.int16)

    # Resample audio to 48000 Hz
    if input_sample_rate != 48000:
        target_sample_rate = 48000
        audio_data = scipy.signal.resample(audio_data, int(len(audio_data) * target_sample_rate / input_sample_rate))

    # If input audio is mono (1 channel), convert to stereo by duplicating channels
    if input_channels == 1:
        audio_data = np.repeat(audio_data, 2)

    return audio_data.astype(np.int16).tobytes()

# Capture and buffer audio in batches
async def capture_mic(vc):
    """Capture microphone input in batches and buffer it for playback."""
    loop = asyncio.get_event_loop()

    while True:
        try:
            # Capture a chunk of audio in a background thread (non-blocking)
            data = await loop.run_in_executor(None, stream.read, CHUNK)

            # Process the audio data to ensure it's 16-bit PCM, 48000 Hz, stereo
            processed_data = process_audio(data)

            # Add the processed audio to the buffer
            audio_buffer.put(processed_data)

        except Exception as e:
            print(f"Error while capturing audio: {e}")
            break

        await asyncio.sleep(0)  # Yield to the event loop to avoid blocking

async def play_buffered_audio(vc):
    """Play buffered audio from the queue."""
    while True:
        try:
            # Check if there is audio in the buffer
            if not audio_buffer.empty():
                # Get audio data from the buffer
                processed_data = audio_buffer.get()

                # Convert processed audio to a file-like object for Discord to play
                audio_stream = io.BytesIO(processed_data)

                # Check if the bot is already playing, if not, start playback
                if not vc.is_playing():
                    vc.play(discord.PCMAudio(audio_stream))

        except Exception as e:
            print(f"Error while playing audio: {e}")
            break

        await asyncio.sleep(0.01)  # Small sleep to yield control to the event loop

async def play_mic(vc):
    """Play microphone input with altered pitch and tone"""
    loop = asyncio.get_event_loop()
    with open("test_audio.raw", "wb") as f:
        while True:
            try:
                # Read audio from mic
                
                
                data = await loop.run_in_executor(None, stream.read, CHUNK)
                
                # data = stream.read(CHUNK)
                save_processed_audio(data, input_sample_rate=44100, input_channels=1, filename="test_audio.pcm")

                altered_data = process_audio(data)

                # Save the raw audio data to a file for testing
                # f.write(altered_data)

                # Use ffmpeg to ensure correct PCM format
                '''
                ffmpeg_process = subprocess.Popen(
                    ['ffmpeg', '-f', 's16le', '-ar', '44100', '-ac', '1', '-i', 'pipe:0', 
                    '-f', 's16le', '-ar', '48000', '-ac', '2', 'pipe:1'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE
                )

                # Pass the altered data to ffmpeg for conversion
                stdout_data, stderr = ffmpeg_process.communicate(input=altered_data)
                '''
                # f.write(altered_data)
                
                # Ensure stdout_data is properly formatted PCM audio for Discord
                audio_stream = io.BytesIO(altered_data)

                # Check if the bot is already playing, and queue the next one if not
                if not vc.is_playing():
                    # Play the audio if the bot isn't already playing
                    print("Bot is starting to play audio")
                    # Create a PCM Audio stream and play it in the voice channel
                    vc.play(discord.PCMAudio(audio_stream))

            except Exception as e:
                print(f"Error: {e}")
                break
            await asyncio.sleep(0)  # Yield control to event loop

bot.run(TOKEN)
