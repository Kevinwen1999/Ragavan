import tempfile
import discord
import asyncio
import edge_tts
import requests
import json
import os
from discord.ext import commands
from dotenv import load_dotenv
from discord.ext import voice_recv
from datetime import datetime
import wave
import io
import threading
import time
import whisper
import numpy as np

if __name__ == '__main__':

    # Load the environment variables from .env file
    load_dotenv()

    # Configuration and constants
    LLM_ENDPOINT = 'http://localhost:1234/v1/chat/completions'  # Update your LLM endpoint if needed
    DISCORD_TOKEN = os.getenv('DISCORD_BOT_TOKEN')  # Replace with your bot token
    COMMAND_PREFIX = "Mario"
    SILENCE_TIMEOUT = 1  # Timeout in seconds to process voice data after the user stops speaking

    # Initialize Discord bot
    bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

    # Initialize Whisper STT model
    model = whisper.load_model("large-v3-turbo")  # Use 'base' model; other models are available

    # Dictionary to keep track of user audio files
    audio_files = {}

    class MySink(voice_recv.AudioSink):
        def __init__(self):
            super().__init__()
            self.is_speaking = False
            self.audio_buffer = bytearray()  # To accumulate audio data
            self.last_speech_time = time.time()  # Track the last time speech was heard
            self.lock = threading.Lock()  # For thread safety with audio buffer
            self.timeout_thread = threading.Thread(target=self.check_silence)
            self.timeout_thread.daemon = True
            self.timeout_thread.start()

        def wants_opus(self) -> bool:
            """Return False to receive PCM (uncompressed) audio, True for Opus."""
            return False  # Set to True if you want Opus instead of PCM

        def write(self, user, data):
            """Process the incoming voice data and accumulate it."""
            pcm_data = data.pcm    
            with self.lock:
                self.audio_buffer.extend(pcm_data)  # Add to buffer
            self.last_speech_time = time.time()  # Reset silence timer
            # print(f"Received PCM data: {len(pcm_data)} bytes")

        def check_silence(self):
            """Continuously check for silence and process the audio data."""
            while True:
                time.sleep(0.5)  # Check every 500ms
                if time.time() - self.last_speech_time > SILENCE_TIMEOUT:
                    # If silence timeout is reached, process the accumulated audio
                    if self.audio_buffer:
                        print("No speech detected for a while, processing audio...")
                        self.process_audio()

        def process_audio(self):
            """Actual audio processing for Whisper in a non-blocking way."""
            with self.lock:
                audio_data = self.audio_buffer

            # Save audio to a temporary file for Whisper (requires .wav format)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file_name = tmp_file.name
                self.save_audio_to_wav(tmp_file_name, audio_data)

            # Use Whisper to transcribe the audio file
            result = model.transcribe(tmp_file_name)
            print(f"Transcript: {result['text']}")

            # Handle LLM query based on transcribed text
            text = result['text'].lstrip()
            if text:
                print(f"Voice input: {text}")
                if text.lower().startswith(COMMAND_PREFIX.lower()):
                    query = text[len(COMMAND_PREFIX):].strip()
                    print(f"Received query: {query}")
                    response = send_to_llm(query)
                    if response:
                        print(f"LLM response: {response}")
                        self.play_tts_in_voice_channel(response)

        def save_audio_to_wav(self, file_path, pcm_data):
            """Write the accumulated PCM audio data into a .wav file with the correct format."""
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(2)  # Mono channel
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(48000)  # 16kHz sample rate
                wf.writeframes(pcm_data)  # Write the PCM audio data to the file
                print(f"Audio saved to {file_path}")
            with self.lock:
                self.audio_buffer.clear()  # Clear the buffer after processing
            wf.close()

        def play_tts_in_voice_channel(self, text):
            """Convert the LLM response to TTS and play it in the voice channel."""
            tts = edge_tts.Communicate(text)
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file_name = tmp_file.name
                asyncio.run(tts.save(tmp_file_name))

                # Rewind the file to the beginning before sending it to Discord
                tmp_file.seek(0)

                # Play the audio in the voice channel
                if bot.voice_clients:
                    voice_client = bot.voice_clients[0]
                    voice_client.play(discord.FFmpegPCMAudio(tmp_file_name), after=lambda e: os.remove(tmp_file_name))

        def cleanup(self):
            """Cleanup after recording is done. Close all open files."""
            for user_id, wf in audio_files.items():
                wf.close()
                print(f"Closed audio file for user {user_id}")
            audio_files.clear()

    # Function to send a query to the LLM endpoint
    def send_to_llm(query):
        payload = {
            "messages": [{"role": "user", "content": query}]
        }

        try:
            response = requests.post(
                LLM_ENDPOINT,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
            if response.status_code == 200:
                result = response.json()

                # Extract the LLM response
                try:
                    bot_response = result['choices'][0]['message']['content']
                except (KeyError, IndexError):
                    bot_response = 'Sorry, I didn’t understand that.'

                return bot_response
            else:
                print(f"LLM request failed with status {response.status_code}")
                return None
        except Exception as e:
            print(f"Error sending to LLM: {e}")
            return None

    # Event when bot is ready
    @bot.event
    async def on_ready():
        print(f"Bot logged in as {bot.user}")

    # Event when bot joins a voice channel
    @bot.command()
    async def join(ctx):
        """Join a voice channel and start recording."""
        if ctx.author.voice:
            # Join the voice channel
            voice_channel = ctx.author.voice.channel
            vc = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
            
            # Start listening to audio as soon as the bot joins the channel
            vc.listen(MySink())  # Capture the audio with MySink

    # Command to leave voice channel
    @bot.command()
    async def leave(ctx):
        """Leave the voice channel."""
        if ctx.voice_client:
            ctx.voice_client.stop_listening()
            await ctx.voice_client.disconnect()

    # Run the bot
    bot.run(DISCORD_TOKEN)
