"""
download_clips.py

Downloads talks from a TED playlist (default: "Open Source, Open World")
and trims each into one or more short clips (30-90s) suitable for the
gesture-speech alignment pipeline.

--- IMPORTANT NOTE (2026-09) ---
yt-dlp's TED extractor is currently broken against ted.com's own pages
(see https://github.com/yt-dlp/yt-dlp/issues/17507) — it fails with
"the JSON object must be str, bytes or bytearray, not NoneType" on every
single talk page. This is an upstream bug, not a local setup problem.

To work around it, this script still uses ted.com to LIST the playlist
(that part works fine), but DOWNLOADS each talk from its official TED
YouTube mirror instead of ted.com directly, since yt-dlp's YouTube
extractor is unaffected. 
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

PLAYLIST_URL = "https://www.ted.com/playlists/13/open_source_open_world"

RAW_DIR = Path("data/raw")
CLIPS_DIR = Path("data/clips")

TED_YT_CHANNEL_MATCH = "TED"


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _check_js_runtime() -> None:
    import shutil
    for exe in ("deno", "node", "bun"):
        if shutil.which(exe):
            return
    print(
        "WARNING: no JS runtime (deno/node/bun) found on PATH. "
        "YouTube downloads will likely fail with 'HTTP Error 403: Forbidden'. "
        "Install Deno first:\n"
        "  curl -fsSL https://deno.land/install.sh | sh -s -- -y\n"
        "  export PATH=\"$HOME/.deno/bin:$PATH\"\n"
    )


def slug_to_query(slug_or_url: str) -> str:
    slug = slug_or_url.rstrip("/").split("/")[-1]
    slug = re.sub(r"[_\-]+", " ", slug).strip()
    return slug


def list_playlist(playlist_url: str) -> list[dict]:
    result = subprocess.run(
        ["yt-dlp", "--flat-playlist", "-J", playlist_url],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    entries = data.get("entries", []) or []
    videos = []
    for e in entries:
        url = e.get("url") or e.get("webpage_url") or e.get("id")
        title = e.get("title")
        query = title if title else slug_to_query(url)
        videos.append({"id": e.get("id"), "title": title, "query": query, "ted_url": url})
    return videos


def find_youtube_match(query: str, cookies_file: str | None = None) -> str | None:
    search_query = f"ytsearch5:{query} TED"
    cmd = ["yt-dlp", "--flat-playlist", "-J"]
    if cookies_file:
        cmd += ["--cookies", cookies_file]
    cmd.append(search_query)
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout:
        return None

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    entries = data.get("entries", []) or []
    if not entries:
        return None

    for e in entries:
        uploader = (e.get("uploader") or e.get("channel") or "").strip()
        if uploader.lower() == "ted":
            return e.get("url") or e.get("webpage_url")


    top = entries[0]
    return top.get("url") or top.get("webpage_url")


def download_video(query: str, out_dir: Path, cookies_file: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    youtube_url = find_youtube_match(query, cookies_file=cookies_file)
    if youtube_url is None:
        raise RuntimeError(f"No YouTube match found for query: {query!r}")

    print(f"  matched YouTube URL: {youtube_url}")

    out_template = str(out_dir / "%(title).80s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", out_template,
        "--restrict-filenames",
    ]
    if cookies_file:
        cmd += ["--cookies", cookies_file]
    cmd.append(youtube_url)
    run(cmd)
    mp4s = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    return mp4s[-1]

def trim_clip(src: Path, start: float, duration: float, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src.stem}_clip_{int(start)}s-{int(start+duration)}s.mp4"
    run([
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(src),
        "-t", str(duration),
        "-c:v", "libx264", "-c:a", "aac",
        str(out_path),
    ])
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--playlist-url", default=PLAYLIST_URL, help="TED playlist URL")
    ap.add_argument("--list-only", action="store_true", help="Just print playlist contents and exit")
    ap.add_argument("--max-videos", type=int, default=8, help="How many talks to download")
    ap.add_argument("--trim-start", type=float, default=30.0, help="Seconds into the talk to start the clip")
    ap.add_argument("--trim-duration", type=float, default=60.0, help="Clip duration in seconds (30-90 recommended)")
    ap.add_argument("--no-trim", action="store_true", help="Download full talks only, skip trimming")
    ap.add_argument("--trim-only", type=str, default=None, help="Path to an already-downloaded video to trim (skips download)")
    ap.add_argument(
        "--cookies-file",
        type=str,
        default=None,
        help=(
            "Path to a Netscape-format cookies.txt exported from a logged-in "
            "browser (e.g. via the 'Get cookies.txt LOCALLY' Chrome extension). "
            "Needed if yt-dlp reports 'Sign in to confirm you're not a bot'."
        ),
    )
    args = ap.parse_args()

    _check_js_runtime()

    if args.trim_only:
        src = Path(args.trim_only)
        out = trim_clip(src, args.trim_start, args.trim_duration, CLIPS_DIR)
        print(f"Saved clip: {out}")
        return

    videos = list_playlist(args.playlist_url)
    print(f"Found {len(videos)} videos in playlist.")
    for i, v in enumerate(videos):
        print(f"  [{i}] {v['query']}")

    if args.list_only:
        return

    selected = videos[: args.max_videos]
    for v in selected:
        print(f"\nDownloading: {v['query']}")
        try:
            video_path = download_video(v["query"], RAW_DIR, cookies_file=args.cookies_file)
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"  ! Failed to download {v['query']}, skipping. ({e})")
            continue

        if not args.no_trim:
            try:
                clip_path = trim_clip(video_path, args.trim_start, args.trim_duration, CLIPS_DIR)
                print(f"  -> Clip saved: {clip_path}")
            except subprocess.CalledProcessError:
                print(f"  ! Failed to trim {video_path.name}, skipping trim (full video still saved).")


if __name__ == "__main__":
    main()
