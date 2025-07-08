#!/usr/bin/env python3
import asyncio
import os
import discord
from discord.ext import commands
import config
import logging
from utils.discord_log_handler import DiscordLogHandler
from utils.speeds import preload_dependencies

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(name)s: %(message)s',
    handlers=[
        logging.FileHandler("bot_activity.log"),
        logging.StreamHandler()
    ]
)

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

bot = commands.Bot(command_prefix='?', intents=intents, owner_id=config.BOT_OWNER_ID)

discord_log_handler = None

@bot.event
async def on_ready():
    global discord_log_handler
    logging.info(f'Logged in as {bot.user} (ID: {bot.user.id})')
    logging.info('------')
    if config.LOG_CHANNEL_ID and config.LOG_CHANNEL_ID != "YOUR_LOG_CHANNEL_ID":
        discord_log_handler = DiscordLogHandler(bot, config.LOG_CHANNEL_ID)
        logging.getLogger().addHandler(discord_log_handler)
        logging.info(f"Discord log handler added for channel ID: {config.LOG_CHANNEL_ID}")
    else:
        logging.warning("LOG_CHANNEL_ID is not set. Discord logging will be disabled.")

async def load_extensions():
    """Loads all cogs from the cogs and utils directories."""
    # Explicitly list the cogs to load
    cogs_to_load = [
        'cogs.admin',
        'cogs.queuebuffer',
        'cogs.nsfw',
        'cogs.cleaner',
        'cogs.neural_network_cog', # Load the new AI cog
        'cogs.log_cog',
        'cogs.youtube',
        'cogs.meme',
        # 'cogs.music', # Removed to prevent duplicate loading
        'utils.self_healing', # Self-healing is also treated as a cog
    ]
    for extension in cogs_to_load:
        try:
            await bot.load_extension(extension)
            logging.info(f'Successfully loaded extension: {extension}')
        except Exception as e:
            logging.error(f'Failed to load extension {extension}: {e}', exc_info=True)

async def main():
    preload_dependencies()
    os.makedirs("audio_cache", exist_ok=True)
    os.makedirs("yt_dlp_cache", exist_ok=True)
    logging.info("Checked and ensured cache directories exist.")

    async with bot:
        await load_extensions()
        try:
            await bot.start(config.DISCORD_TOKEN)
        except discord.errors.LoginFailure:
            logging.error("Error: Invalid Discord Token. Please check your DISCORD_TOKEN in config.py.")
        except Exception as e:
            logging.error(f"Error when starting bot: {e}", exc_info=True)

@bot.command()
async def reload(ctx, extension):
    """Reloads an extension."""
    try:
        await bot.unload_extension(extension)
        await bot.load_extension(extension)
        await ctx.send(f"Extension {extension} reloaded.")
    except Exception as e:
        await ctx.send(f"Error reloading extension {extension}: {e}")
# Remove the custom help command to avoid conflict with the default help command
# @bot.command(name='help', help='Lists all available commands.')
# async def help_command(ctx):
#     """Lists all available commands."""
#     command_list = []
#     for command in bot.commands:
#         command_list.append(f"**{config.COMMAND_PREFIX}{command.name}** - {command.help or 'No description provided.'}")

#     if command_list:
#         help_message = "Here are the available commands:\n" + "\n".join(command_list)
#     else:
#         help_message = "No commands found."

#     await ctx.send(help_message)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped.")
