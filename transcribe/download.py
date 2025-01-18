import re
import tempfile
from pathlib import Path
import yt_dlp
from typing import Optional

def is_url(path: str) -> bool:
    """Check if path is a URL."""
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(path) is not None

def download_media(url: str, output_dir: Optional[Path] = None) -> Path:
    """Download media from URL using yt-dlp."""
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    
    ydl_opts = {
        'format': 'best',
        'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info))