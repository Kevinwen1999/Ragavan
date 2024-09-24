import os
import discord
import aiohttp
import json
import tiktoken

# Load your bot token
TOKEN = ''

# LLM API endpoint
LLM_ENDPOINT = 'http://localhost:1234/v1/chat/completions'

# Path to the file where user message histories will be saved
MESSAGE_HISTORY_FILE = 'user_message_histories.json'

# Maximum number of tokens to maintain for context (you can adjust based on your model's limits)
MAX_TOKENS = 7000  # Adjust according to the model’s total token limit
MAX_MESSAGES_HISTORY = 100  # Max number of messages to keep in history

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Set the bot's intents (permissions)
intents = discord.Intents.default()
intents.message_content = True

# Create a bot instance
client = discord.Client(intents=intents)

# Initialize dictionaries to store conversation history and last input for each user
conversation_histories = {}
last_user_inputs = {}

# Estimate the token count by counting the words in a message
def estimate_token_count(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

# Function to split long messages into chunks of 2000 characters
def split_message(message, max_length=2000):
    return [message[i:i+max_length] for i in range(0, len(message), max_length)]

def trim_messages_to_fit(messages, max_tokens):
    token_count = sum(estimate_token_count(msg) for msg in messages)
    
    # Trim the oldest messages if the total exceeds the max_tokens limit
    while token_count > max_tokens and len(messages) > 1:
        token_count -= estimate_token_count(messages.pop(0))
    
    return messages

# Function to trim conversation history to fit within token limits
def trim_conversation_history(history, max_tokens):
    token_count = sum(estimate_token_count(message['content']) for message in history)
    while token_count > max_tokens and len(history) > 1:
        token_count -= len(history[0]['content'])
        history.pop(0)
    return history

# Load existing user message histories from file (if it exists)
def load_message_histories():
    if os.path.exists(MESSAGE_HISTORY_FILE):
        with open(MESSAGE_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return {}

# Save user message histories to a file
def save_message_histories(histories):
    with open(MESSAGE_HISTORY_FILE, 'w') as f:
        json.dump(histories, f)

# Initialize user message histories
user_message_histories = load_message_histories()

# Function to collect user messages and save to the file
def collect_user_message(user_id, message):
    if user_id not in user_message_histories:
        user_message_histories[user_id] = []
    user_message_histories[user_id].append(message)

    # Save to file after collecting a new message
    save_message_histories(user_message_histories)

# Function to inject a user's chat history from Discord
async def inject_user_chat_history(message, target_user, limit=50):
    target_user_id = str(target_user.id)
    # Fetch message history from the channel
    injected_messages = []

    async for msg in message.channel.history(limit=limit):
        if msg.author == target_user:  # Only include messages from the target user
            injected_messages.append(msg.content)

    if not injected_messages:
        await message.channel.send(f"No messages found for {target_user.name}.")
        return

    # Inject messages into the user's message history
    if target_user_id not in user_message_histories:
        user_message_histories[target_user_id] = []

    user_message_histories[target_user_id].extend(injected_messages)

    # Save the updated message history to the file
    save_message_histories(user_message_histories)

    await message.channel.send(f"Injected {len(injected_messages)} messages from {target_user.name} into their message history.")

# Function to generate a personality and sentiment analysis of the user
async def analyze_user(message, target_user_id):
    # Collect the user's messages and trim to fit under the MAX_TOKENS limit
    user_messages = user_message_histories.get(target_user_id, [])

    if not user_messages or len(user_messages) < 1:
        await message.channel.send("Not enough messages to analyze this user's style.")
        return

    # Trim messages to fit within MAX_TOKENS
    trimmed_messages = trim_messages_to_fit(user_messages, MAX_TOKENS)

    # Craft the prompt to analyze the target user's personality and sentiment
    prompt = "Analyze the following user's personality and sentiment based on their recent messages:\n\n"
    for user_message in trimmed_messages:
        prompt += f"- {user_message}\n"

    prompt += "\nProvide a detailed analysis of their personality traits and overall sentiment."

    payload = {
        "messages": [{"role": "user", "content": prompt}]
    }

    async with message.channel.typing():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(LLM_ENDPOINT, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()

                        # Extract the LLM response
                        try:
                            bot_response = result['choices'][0]['message']['content']
                        except (KeyError, IndexError):
                            bot_response = 'Sorry, I didn’t understand that.'

                        # Split and send the response if it's longer than 2000 characters
                        messages_to_send = split_message(bot_response)
                        for chunk in messages_to_send:
                            await message.channel.send(chunk)
                    else:
                        await message.channel.send('Failed to reach the LLM endpoint.')
        except Exception as e:
            await message.channel.send(f"An error occurred: {str(e)}")

# Function to generate a response imitating the user's style
async def imitate_user_style(message, target_user_id):
    # Collect the last N messages from the user to use as examples
    user_messages = user_message_histories.get(target_user_id, [])

    if not user_messages or len(user_messages) < 1:
        await message.channel.send("Not enough messages to imitate this user's style.")
        return


    # Trim messages to fit within MAX_TOKENS
    trimmed_messages = trim_messages_to_fit(user_messages, MAX_TOKENS)

    # Craft the prompt to imitate the target user's style
    prompt = "Generate one sentence by imitating the following user's style of speech. Here are some of their recent messages:\n\n"
    for user_message in trimmed_messages:  # Use the last 5 messages as examples
        prompt += f"- {user_message}\n"

    prompt += "\nGenerate one sentence that imitates this user's style of speech."

    payload = {
        "messages": [{"role": "user", "content": prompt}]
    }

    async with message.channel.typing():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(LLM_ENDPOINT, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()

                        # Extract the LLM response
                        try:
                            bot_response = result['choices'][0]['message']['content']
                        except (KeyError, IndexError):
                            bot_response = 'Sorry, I didn’t understand that.'

                        # Split and send the response if it's longer than 2000 characters
                        messages_to_send = split_message(bot_response)
                        for chunk in messages_to_send:
                            await message.channel.send(chunk)
                    else:
                        await message.channel.send('Failed to reach the LLM endpoint.')
        except Exception as e:
            await message.channel.send(f"An error occurred: {str(e)}")

# Function to generate a response using LLM (actual API call)
async def generate_response(message, user_id, continue_generation=False):
    # Prepare the payload for the LLM API (including conversation history)
    payload = {
        "messages": conversation_histories[user_id]
    }

    # Modify the message if it's a continuation
    if continue_generation:
        payload["messages"].append({
            "role": "user",
            "content": "Please continue."
        })

    async with message.channel.typing():
        try:
            # Make the API call to the LLM endpoint
            async with aiohttp.ClientSession() as session:
                async with session.post(LLM_ENDPOINT, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()

                        # Extract the assistant's response from the "choices" field
                        try:
                            bot_response = result['choices'][0]['message']['content']
                        except (KeyError, IndexError):
                            bot_response = 'Sorry, I didn’t understand that.'

                        # Add the bot's response to the conversation history
                        conversation_histories[user_id].append({
                            "role": "assistant",
                            "content": bot_response
                        })

                        # Split the response if it's longer than 2000 characters
                        messages_to_send = split_message(bot_response)

                        # Send the chunks of messages one by one
                        for chunk in messages_to_send:
                            await message.channel.send(chunk)
                    else:
                        await message.channel.send('Failed to reach the LLM endpoint.')
        except Exception as e:
            await message.channel.send(f"An error occurred: {str(e)}")

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

# Handle incoming messages
@client.event
async def on_message(message):
    # Prevent the bot from replying to itself
    if message.author == client.user:
        return

    # Get the user's ID to maintain separate conversation history for each user
    user_id = str(message.author.id)

    # Initialize conversation history for the user if it doesn't exist
    if user_id not in conversation_histories:
        conversation_histories[user_id] = []

    # Command to inject a user's chat history
    if message.content.startswith('!inject'):
        # Extract the mentioned user from the message and the limit
        if message.mentions:
            target_user = message.mentions[0]
            try:
                # Try to extract the limit from the message (e.g., !inject @user_name 100)
                limit = int(message.content.split()[-1])
            except (IndexError, ValueError):
                limit = 50  # Default limit if not specified

            await inject_user_chat_history(message, target_user, limit=limit)
        else:
            await message.channel.send("Please mention a user to inject their chat history. Usage: `!inject @user_name <limit>`")
        return
    
    if message.content.startswith('!analyze'):
        # Extract the mentioned user from the message
        if message.mentions:
            target_user = message.mentions[0]
            target_user_id = str(target_user.id)
            await analyze_user(message, target_user_id)
        else:
            await message.channel.send("Please mention a user to analyze. Usage: `!analyze @user_name`")
        return

    # Command to imitate another user's style
    if message.content.startswith('!imitate'):
        # Extract the mentioned user from the message
        if message.mentions:
            target_user = message.mentions[0]
            target_user_id = str(target_user.id)
            await imitate_user_style(message, target_user_id)
        else:
            await message.channel.send("Please mention a user to imitate. Usage: `!imitate @user_name`")
        return

    # If the user sends the reset command, clear their conversation history
    if message.content.startswith('!reset'):
        conversation_histories[user_id] = []
        last_user_inputs[user_id] = None
        await message.channel.send("Your conversation has been reset.")
        return

    # If the user sends the continue command
    if message.content.startswith('!continue'):
        if last_user_inputs.get(user_id):
            # Continue generating the response based on the last input
            await generate_response(message, user_id, continue_generation=True)
        else:
            await message.channel.send("No previous input found to continue.")
        return

    # If the user sends the regenerate command
    if message.content.startswith('!regenerate'):
        # Check if we have the last user input
        if last_user_inputs.get(user_id):
            # Remove the last assistant response (if it exists) from the conversation history
            if conversation_histories[user_id]:
                last_message = conversation_histories[user_id][-1]
                if last_message['role'] == 'assistant':
                    conversation_histories[user_id].pop()

            # Regenerate response based on last input
            await generate_response(message, user_id)
        else:
            await message.channel.send("No previous input found to regenerate.")
        return

    # If someone sends a message starting with !ask
    if message.content.startswith('!ask'):
        # Extract the user query after the command prefix
        user_input = message.content[len('!ask '):]

        # Add the new user message to the conversation history
        conversation_histories[user_id].append({
            "role": "user",
            "content": user_input
        })

        # Store the last user input separately for the regenerate and continue commands
        last_user_inputs[user_id] = user_input

        # Trim the conversation history to fit within the token limit
        conversation_histories[user_id] = trim_conversation_history(conversation_histories[user_id], MAX_TOKENS)

        # Generate the response
        await generate_response(message, user_id)

    # Handle the help command
    if message.content.startswith('!help'):
        help_message = """
        **Available Commands:**
        - `!ask <your question>`: Ask the bot a question.
        - `!continue`: Continue the last response based on previous input.
        - `!regenerate`: Regenerate the bot's response to the last question.
        - `!reset`: Reset the conversation and start fresh.
        - `!inject @user_name <limit>`: Inject a user’s chat history (up to <limit> messages) from this channel into their message history.
        - `!imitate @user_name`: Imitate the style of the mentioned user based on their past messages.
        - `!analyze @user_name`: Do a personality and sentiment analysis of the mentioned user based on their past messages.
        - `!help`: Show this help message.
        """
        await message.channel.send(help_message)
        return
    # Collect the user's message for imitation purposes
    collect_user_message(user_id, message.content)

# Run the bot
client.run(TOKEN)