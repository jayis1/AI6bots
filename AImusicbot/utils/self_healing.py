import asyncio
import logging
import os
import sys
from discord.ext import tasks, commands
import config
from .db_utils import log_healing_event, initialize_db

class SelfHealing(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        initialize_db()
        self.health_check.start()

    def cog_unload(self):
        self.health_check.cancel()

    @tasks.loop(seconds=60)
    async def health_check(self):
        # Check if the bot is closed
        if self.bot.is_closed():
            log_healing_event("Bot disconnected", "Attempting to reconnect.")
            try:
                await self.bot.login(config.DISCORD_TOKEN)
                await self.bot.connect()
                log_healing_event("Reconnected successfully.")
            except Exception as e:
                log_healing_event("Reconnect failed", str(e))
                self.restart_bot()

        # Check the Discord gateway latency
        if self.bot.latency > 1.0:
            log_healing_event("High latency detected", f"Latency: {self.bot.latency}")
            await self.bot.close()
            await self.bot.login(config.DISCORD_TOKEN)
            await self.bot.connect()
            log_healing_event("Reconnected due to high latency.")

    def restart_bot(self):
        log_healing_event("Restarting bot")
        os.execv(sys.executable, ['python'] + sys.argv)

    def generate_error_summary(self, error_message):
        # Get the NeuralNetworkCog instance
        neural_network_cog = self.bot.get_cog("NeuralNetworkCog")
        if not neural_network_cog or not neural_network_cog.ai_manager.is_ready():
            return "Local AI model not available. Cannot generate error summary."
        
        prompt = f"Summarize the following Python error and suggest a potential cause:\n\n{error_message}\n\nSummary:"
        try:
            # Use the process_command method from the AIModelManager via the NeuralNetworkCog
            # Use the 'summarize' command for generating error summaries
            return neural_network_cog.ai_manager.process_command('summarize', f"{prompt}\n\n{error_message}")
        except Exception as e:
            logging.error(f"Error generating summary with local AI: {e}", exc_info=True)
            return "Failed to generate error summary."

    @commands.Cog.listener()
    async def on_command_error(self, ctx, error):
        log_healing_event("Command error", f"Command: {ctx.command}, Error: {error}")
        
        if isinstance(error, commands.CommandNotFound):
            return # Don't respond to invalid commands

        summary = self.generate_error_summary(str(error))

        response = (
            f"I've encountered an error in the `{ctx.command}` command.\n\n"
            f"**AI-Generated Summary:**\n"
            f"```\n{summary}\n```\n"
            f"Please check the logs for the full traceback."
        )
        
        try:
            await ctx.send(response)
        except Exception as e:
            logging.error(f"Failed to send error message to channel: {e}")

async def setup(bot):
    await bot.add_cog(SelfHealing(bot))
