from datetime import datetime
import os
import wave
import discord
from discord.ext import commands, voice_recv
import asyncio
import numpy as np
from discord.ext import commands
from dotenv import load_dotenv

import moondream as md
from PIL import Image

# Initialize with local model path. Can also read .mf.gz files, but we recommend decompressing
# up-front to avoid decompression overhead every time the model is initialized.
model = md.vl(model="./moondream-2b-int8.mf")

# Load the environment variables from .env file
load_dotenv()

# Get the Discord bot token from .env
TOKEN = os.getenv('DISCORD_BOT_TOKEN')


bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())


# We'll store the path to the current (last) image file here.
current_image_path = None

@bot.event
async def on_ready():
    print(f"Bot is online as {bot.user}")

@bot.command(name="img")
async def generate_or_modify_image(ctx, *, prompt=None):
    """
    Usage:
      !img [prompt text]
    Attach an image to use as a starting point (init_image).
    If no attachment is provided, uses the last image if available.
    """
    global current_image_path

    # Grab the first attachment if any
    attachment = ctx.message.attachments[0] if ctx.message.attachments else None

    # If a new image is attached, download & store it
    if attachment:
        input_filename = "input.png"
        await attachment.save(input_filename)
        current_image_path = input_filename
    else:
        # If there's no attachment and we don't have a stored image, we can't proceed
        if not current_image_path:
            await ctx.send("No new image attached and no previous image stored.")
            return

    # Use a default prompt if none was provided
    if prompt is None:
        prompt = "A mystic dreamlike landscape"

    image = Image.open("./input.png")
    encoded_image = model.encode_image(image)
    # caption = model.caption(encoded_image)["caption"]
    answer = model.query(encoded_image, prompt)["answer"]

    await ctx.send(answer)

@bot.command(name="imgC")
async def generate_or_modify_image(ctx, *, prompt=None):
    """
    Usage:
      !img [prompt text]
    Attach an image to use as a starting point (init_image).
    If no attachment is provided, uses the last image if available.
    """
    global current_image_path

    # Grab the first attachment if any
    attachment = ctx.message.attachments[0] if ctx.message.attachments else None

    # If a new image is attached, download & store it
    if attachment:
        input_filename = "input.png"
        await attachment.save(input_filename)
        current_image_path = input_filename
    else:
        # If there's no attachment and we don't have a stored image, we can't proceed
        if not current_image_path:
            await ctx.send("No new image attached and no previous image stored.")
            return

    # Use a default prompt if none was provided
    if prompt is None:
        prompt = "A mystic dreamlike landscape"

    image = Image.open("./input.png")
    encoded_image = model.encode_image(image)
    # caption = model.caption(encoded_image)["caption"]
    answer = model.caption(encoded_image)["caption"]

    await ctx.send(answer)


@bot.command(name="showCurrentImg")
async def show_current_image(ctx):
    """
    Shows the currently stored/generated image if it exists.
    """
    global current_image_path
    if current_image_path and os.path.isfile(current_image_path):
        await ctx.send(file=discord.File(current_image_path))
    else:
        await ctx.send("No image is currently stored.")

# Replace with your bot token
bot.run(TOKEN)