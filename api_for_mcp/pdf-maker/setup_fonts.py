"""Download DejaVu fonts for Unicode/Cyrillic support."""
import urllib.request
from pathlib import Path

FONTS_DIR = Path(__file__).parent / "fonts"
FONTS_DIR.mkdir(exist_ok=True)

FONTS = {
    "DejaVuSans.ttf": "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf": "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Bold.ttf",
}

for filename, url in FONTS.items():
    dest = FONTS_DIR / filename
    if dest.exists():
        print(f"  {filename} already exists, skipping.")
        continue
    print(f"  Downloading {filename}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")

print("Done.")
