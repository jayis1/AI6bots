import asyncio
import functools
import logging
import yt_dlp
import discord
import os

# Suppress noise from yt-dlp
yt_dlp.utils.bug_reports_hook = lambda *args, **kwargs: None

# Configure logging for yt-dlp
class YTDLLogger:
    def debug(self, msg):
        # Log debug messages from yt-dlp if needed, but often too verbose
        # logging.debug(f"YTDL: {msg}")
        pass

    def warning(self, msg):
        logging.warning(f"YTDL: {msg}")

    def error(self, msg):
        logging.error(f"YTDL: {msg}")

# FFmpeg options for playing audio
FFMPEG_OPTIONS = {
    'options': '-vn',  # No video
    'before_options': '-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5' # Options to reconnect
}

# YTDL options for extracting audio information
YTDL_FORMAT_OPTIONS = {
    'format': 'bestaudio/best',  # Get the best audio format
    'outtmpl': '%(extractor)s-%(id)s-%(title)s.%(ext)s', # Output template
    'restrictfilenames': True, # Restrict filenames to ASCII
    'noplaylist': True, # Default to not downloading playlists
    'nocheckcertificate': True, # Ignore SSL certificate errors
    'ignoreerrors': False, # Do not ignore errors
    'logtostderr': False, # Do not log to stderr
    'quiet': True, # Suppress console output
    'no_warnings': True, # Suppress warnings
    'default_search': 'ytsearch', # Default search prefix
    'source_address': '0.0.0.0', # Bind to IPv4 since IPv6 often causes issues
    'cookiefile': 'youtube_cookie.txt' if os.path.exists('youtube_cookie.txt') else None, # Use cookie file if it exists
    'logger': YTDLLogger(), # Use custom logger
    'http_headers': { # Add headers to mimic a browser
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36)'
    },
    'extractor_args': { # Use web player client for YouTube
        'youtube': {
            'player_client': ['web']
        }
    }
}

class YTDLSource(discord.PCMVolumeTransformer):
    def __init__(self, source, *, data, volume=0.5):
        super().__init__(source, volume)
        self.data = data
        self.title = data.get('title')
        self.url = data.get('url')
        self.webpage_url = data.get('webpage_url')
        self.duration = data.get('duration')
        self.thumbnail = data.get('thumbnail')

    @classmethod
    async def from_url(cls, url, *, loop=None, stream=False, ytdl_opts=None):
        loop = loop or asyncio.get_event_loop()
        
        # Ensure ytdl_opts is a dictionary
        if ytdl_opts is None:
            ytdl_opts = YTDL_FORMAT_OPTIONS.copy()
        else:
            # Create a copy to avoid modifying the original dictionary
            ytdl_opts = ytdl_opts.copy()

        # If streaming, set the output template to '-' for stdout
        if stream:
            ytdl_opts['outtmpl'] = '-'
            ytdl_opts['noplaylist'] = True # Ensure only single video is processed when streaming

        ydl = yt_dlp.YoutubeDL(ytdl_opts)

        # Use extract_info to get video data without downloading
        # run_in_executor is used to run blocking code in a separate thread
        data = await loop.run_in_executor(None, lambda: ydl.extract_info(url, download=False))

        if 'entries' in data:
            # It's a playlist or a search result with multiple entries
            # Return a list of dictionaries containing data and stream flag
            return [{'data': entry, 'stream': stream} for entry in data['entries']]
        else:
            # It's a single video
            # Return a dictionary containing data and stream flag
            return {'data': data, 'stream': stream}


async def setup(bot):
    try:
        from .music import Music # Import Music cog here to avoid circular dependency
        await bot.add_cog(Music(bot))
    except Exception as e:
        logging.error(f"Failed to load music cog: {e}", exc_info=True)
