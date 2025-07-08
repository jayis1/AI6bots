from discord.ext import tasks, commands
from utils.cleaner import clean_audio_cache as clean_audio_cache_util

class Cleaner(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.clean_audio_cache.start()

    def cog_unload(self):
        self.clean_audio_cache.cancel()

    @tasks.loop(hours=24)
    async def clean_audio_cache(self):
        """
        Periodically cleans the audio cache directory.
        """
        clean_audio_cache_util()

    @clean_audio_cache.before_loop
    async def before_clean_audio_cache(self):
        await self.bot.wait_until_ready()

async def setup(bot):
    await bot.add_cog(Cleaner(bot))
