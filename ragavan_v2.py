import os
import discord
import aiohttp
import json
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from PyPDF2 import PdfReader
import tiktoken

# Load the environment variables from .env file
load_dotenv()

# Get the Discord bot token from .env
TOKEN = os.getenv('DISCORD_BOT_TOKEN')
LLM_ENDPOINT = 'http://localhost:1234/v1/chat/completions'
MESSAGE_HISTORY_FILE = 'user_message_histories.json'
MAX_TOKENS = 7000
MAX_MESSAGES_HISTORY = 100

# Set the bot's intents
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# tokenizer used to calculate token count
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


# --------------- Helper Classes ---------------

class MessageHistoryManager:
    def __init__(self):
        self.message_histories = self.load_message_histories()

    def load_message_histories(self):
        if os.path.exists(MESSAGE_HISTORY_FILE):
            with open(MESSAGE_HISTORY_FILE, 'r') as f:
                return json.load(f)
        return {}

    def save_message_histories(self):
        with open(MESSAGE_HISTORY_FILE, 'w') as f:
            json.dump(self.message_histories, f)

    def collect_user_message(self, user_id, message):
        if user_id not in self.message_histories:
            self.message_histories[user_id] = []
        self.message_histories[user_id].append(message)
        self.save_message_histories()

    def get_user_history(self, user_id):
        return self.message_histories.get(user_id, [])

    def update_user_history(self, user_id, new_history):
        self.message_histories[user_id] = new_history
        self.save_message_histories()


class LLMHandler:
    def __init__(self, endpoint):
        self.endpoint = endpoint

    async def generate_response(self, message, conversation_history, prompt=None):
        payload = {"messages": conversation_history}
        if prompt:
            payload["messages"].append({"role": "user", "content": prompt})

        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content']
                return None

    async def process_with_search_results(self, message, user_query, search_results):
        prompt = f"User asked: {user_query}\n\nHere is some information from a web search:\n{search_results}\nPlease generate a response."
        return await self.generate_response(message, [], prompt)


class Utility:
    @staticmethod
    def estimate_token_count(text):
        return len(tokenizer.encode(text))

    @staticmethod
    def trim_conversation_history(history, max_tokens):
        token_count = sum(Utility.estimate_token_count(message['content']) for message in history)
        while token_count > max_tokens and len(history) > 1:
            token_count -= Utility.estimate_token_count(history.pop(0)['content'])
        return history

    @staticmethod
    def split_message(message, max_length=2000):
        return [message[i:i + max_length] for i in range(0, len(message), max_length)]


class SearchService:
    @staticmethod
    def perform_search(query):
        return DDGS().text(keywords=query, max_results=10, safesearch="off")

    @staticmethod
    def format_search_results(results):
        formatted = ""
        for result in results:
            formatted += f"Title: {result['title']}\nSnippet: {result['body']}\nLink: {result['href']}\n\n"
        return formatted


class PDFHandler:
    @staticmethod
    def extract_text_from_pdf(file_bytes):
        reader = PdfReader(file_bytes)

        if reader.is_encrypted:
            try:
                reader.decrypt('')  # Attempt to decrypt with no password (empty string)
            except Exception:
                raise ValueError("The PDF is encrypted and requires a password.")

        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text


# --------------- Event Handlers ---------------

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    user_id = str(message.author.id)
    message_manager = MessageHistoryManager()
    llm_handler = LLMHandler(LLM_ENDPOINT)

    if message.content.startswith('!ask'):
        await handle_ask_command(message, user_id, message_manager, llm_handler)

    elif message.content.startswith('!net'):
        await handle_net_command(message, user_id, message_manager, llm_handler)

    elif message.content.startswith('!file'):
        await handle_file_command(message, user_id, message_manager, llm_handler)

    elif message.content.startswith('!inject'):
        await handle_inject_command(message)

    elif message.content.startswith('!imitate'):
        await handle_imitate_command(message, user_id, llm_handler)

    elif message.content.startswith('!analyze'):
        await handle_analyze_command(message, llm_handler)

    elif message.content.startswith('!continue'):
        await handle_continue_command(message, user_id, message_manager, llm_handler)

    elif message.content.startswith('!regenerate'):
        await handle_regenerate_command(message, user_id, message_manager, llm_handler)

    elif message.content.startswith('!reset'):
        message_manager.update_user_history(user_id, [])
        await message.channel.send("Your conversation has been reset.")

    elif message.content.startswith('!help'):
        await message.channel.send("""
            **Available Commands:**
            - `!ask <your question>`: Ask the bot a question.
            - `!net <your question>`: Ask the bot a question using web search.
            - `!file <your question>`: Ask the bot a question using an attached text/PDF file.
            - `!continue`: Continue the last response.
            - `!regenerate`: Regenerate the bot's response.
            - `!reset`: Reset the conversation.
            - `!inject @user <limit>`: Inject user's chat history into their message history.
            - `!imitate @user [sentence]`: Imitate the mentioned user's style.
            - `!analyze @user`: Analyze the mentioned user's style.
            - `!help`: Show this help message.
        """)


# --------------- Command Handlers ---------------

async def handle_ask_command(message, user_id, message_manager, llm_handler):
    user_input = message.content[len('!ask '):]
    history = message_manager.get_user_history(user_id)
    history.append({"role": "user", "content": user_input})
    history = Utility.trim_conversation_history(history, MAX_TOKENS)

    response = await llm_handler.generate_response(message, history)
    if response:
        message_manager.update_user_history(user_id, history)
        for chunk in Utility.split_message(response):
            await message.channel.send(chunk)


async def handle_net_command(message, user_id, message_manager, llm_handler):
    user_input = message.content[len('!net '):]
    history = message_manager.get_user_history(user_id)
    history.append({"role": "user", "content": user_input})
    history = Utility.trim_conversation_history(history, MAX_TOKENS)

    search_results = SearchService.perform_search(user_input)
    formatted_results = SearchService.format_search_results(search_results)

    if formatted_results:
        response = await llm_handler.process_with_search_results(message, user_input, formatted_results)
        if response:
            message_manager.update_user_history(user_id, history)
            for chunk in Utility.split_message(response):
                await message.channel.send(chunk)


async def handle_file_command(message, user_id, message_manager, llm_handler):
    user_input = message.content[len('!file '):]
    history = message_manager.get_user_history(user_id)
    history.append({"role": "user", "content": user_input})
    history = Utility.trim_conversation_history(history, MAX_TOKENS)

    if message.attachments:
        attachment = message.attachments[0]

        if attachment.filename.endswith('.txt'):
            file_content = await attachment.read()
            file_text = file_content.decode('utf-8')
            await process_file_content(message, user_id, file_text, user_input, history, message_manager, llm_handler)

        elif attachment.filename.endswith('.pdf'):
            file_content = await attachment.read()
            from io import BytesIO
            file_bytes = BytesIO(file_content)
            pdf_text = PDFHandler.extract_text_from_pdf(file_bytes)
            await process_file_content(message, user_id, pdf_text, user_input, history, message_manager, llm_handler)

        else:
            await message.channel.send("Only .txt and .pdf files are supported.")
    else:
        await message.channel.send("Please attach a file to use the !file command.")


async def handle_inject_command(message):
    if message.mentions:
        target_user = message.mentions[0]
        limit = int(message.content.split()[-1]) if message.content.split()[-1].isdigit() else 50
        await inject_user_chat_history(message, target_user, limit)
    else:
        await message.channel.send("Please mention a user to inject their chat history.")


async def handle_imitate_command(message, user_id, llm_handler):
    if message.mentions:
        target_user = message.mentions[0]
        sentence = message.content.split(' ', 2)[2] if len(message.content.split()) > 2 else None
        await imitate_user_style(message, user_id, target_user, sentence, llm_handler)
    else:
        await message.channel.send("Please mention a user to imitate.")


async def handle_analyze_command(message, llm_handler):
    if message.mentions:
        target_user = message.mentions[0]
        await analyze_user(message, target_user, llm_handler)
    else:
        await message.channel.send("Please mention a user to analyze.")


async def handle_continue_command(message, user_id, message_manager, llm_handler):
    history = message_manager.get_user_history(user_id)
    if history:
        response = await llm_handler.generate_response(message, history, "Please continue.")
        if response:
            for chunk in Utility.split_message(response):
                await message.channel.send(chunk)
    else:
        await message.channel.send("No previous input found to continue.")


async def handle_regenerate_command(message, user_id, message_manager, llm_handler):
    history = message_manager.get_user_history(user_id)
    if history and history[-1]['role'] == 'assistant':
        history.pop()
        response = await llm_handler.generate_response(message, history)
        if response:
            message_manager.update_user_history(user_id, history)
            for chunk in Utility.split_message(response):
                await message.channel.send(chunk)
    else:
        await message.channel.send("No previous input found to regenerate.")


async def process_file_content(message, user_id, file_text, user_query, history, message_manager, llm_handler):
    prompt = f"User asked: {user_query}\n\nHere is some information from the attached file:\n{file_text}\nPlease generate a response."
    response = await llm_handler.generate_response(message, history, prompt)
    if response:
        history.append({"role": "user", "content": prompt})
        history = Utility.trim_conversation_history(history, MAX_TOKENS)
        message_manager.update_user_history(user_id, history)
        for chunk in Utility.split_message(response):
            await message.channel.send(chunk)


# Run the bot
client.run(TOKEN)
