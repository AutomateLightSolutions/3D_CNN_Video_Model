"""
Downloads ffmpeg.exe and ffprobe.exe into backend/bin/.
Run once:  python download_ffmpeg.py
"""
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

BIN_DIR = Path(__file__).parent / "bin"

# BtbN auto-build — small essentials Windows 64-bit release
FFMPEG_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
    "ffmpeg-master-latest-win64-gpl.zip"
)

TOOLS = ("ffmpeg.exe", "ffprobe.exe")


def _progress(block_num: int, block_size: int, total_size: int):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "#" * int(pct / 2)
        print(f"\r  [{bar:<50}] {pct:5.1f}%  {downloaded/1_000_000:.1f} MB", end="", flush=True)


def download():
    BIN_DIR.mkdir(parents=True, exist_ok=True)

    already = [t for t in TOOLS if (BIN_DIR / t).exists()]
    if len(already) == len(TOOLS):
        print(f"FFmpeg binaries already present in {BIN_DIR}")
        return

    print(f"Downloading FFmpeg from:\n  {FFMPEG_URL}\n")
    zip_path = BIN_DIR / "_ffmpeg_download.zip"

    try:
        urllib.request.urlretrieve(FFMPEG_URL, zip_path, reporthook=_progress)
        print()  # newline after progress bar
    except Exception as e:
        zip_path.unlink(missing_ok=True)
        print(f"\nDownload failed: {e}")
        print("Manual alternative: download ffmpeg-release-essentials.zip from")
        print("  https://www.gyan.dev/ffmpeg/builds/")
        print(f"and copy ffmpeg.exe + ffprobe.exe into:  {BIN_DIR}")
        sys.exit(1)

    print("Extracting ffmpeg.exe and ffprobe.exe …")
    found = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if name in TOOLS:
                data = zf.read(member)
                dest = BIN_DIR / name
                dest.write_bytes(data)
                found.append(name)
                print(f"  ✓  {name}  ({len(data)/1_000_000:.1f} MB)")

    zip_path.unlink(missing_ok=True)

    missing = set(TOOLS) - set(found)
    if missing:
        print(f"\nWARN: did not find {missing} inside the zip.")
        sys.exit(1)

    print(f"\nDone. Binaries saved to: {BIN_DIR}")


if __name__ == "__main__":
    download()
