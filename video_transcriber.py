import os
import json
from faster_whisper import WhisperModel
import yt_dlp
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    DownloadColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from pydub import AudioSegment
import tempfile
import re
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
import jiwer
import webvtt

console = Console()

CHUNK_SIZE = 30  # seconds (longer chunk for better alignment)


def clean_text(text):
    """Clean and normalize text for comparison."""
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Handle special characters and symbols
    replacements = {
        "&": "and",  # Replace ampersand
        "Â": "",  # Remove special space character
        "…": "...",  # Replace ellipsis
        "–": "-",  # Replace en dash
        "—": "-",  # Replace em dash
        "″": "",  # Remove double prime
        "′": "",  # Remove prime
        "„": "",  # Remove low double quote
        "‟": "",  # Remove high double quote
        "‚": "",  # Remove low single quote
        "‛": "",  # Remove high single quote
        "«": "",  # Remove left double angle quote
        "»": "",  # Remove right double angle quote
        "‹": "",  # Remove left single angle quote
        "›": "",  # Remove right single angle quote
        ".": " dot ",  # Replace dot with spoken form
        "/": " slash ",  # Replace slash with spoken form
        "-": " dash ",  # Replace dash with spoken form
        "_": " underscore ",  # Replace underscore with spoken form
        "@": " at ",  # Replace at symbol with spoken form
        "#": " hash ",  # Replace hash with spoken form
        "+": " plus ",  # Replace plus with spoken form
        "=": " equals ",  # Replace equals with spoken form
        "?": " question mark ",  # Replace question mark with spoken form
        "!": " exclamation mark ",  # Replace exclamation mark with spoken form
        "'": "",  # Remove single quote
        '"': "",  # Remove double quote
        ",": "",  # Remove comma
    }

    # Apply replacements
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Handle URLs and domains
    text = re.sub(r"([a-z0-9]+)\.(com|org|net)", r"\1 dot \2", text)

    # Remove any remaining special characters except alphanumeric, spaces, and 'dot' (periods are already replaced)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove common filler words
    filler_words = {
        "um",
        "uh",
        "ah",
        "er",
        "like",
        "you know",
        "i mean",
        "well",
        "so",
        "basically",
        "actually",
        "literally",
        "just",
        "kind of",
        "sort of",
        "you see",
    }
    words = text.split()
    words = [w for w in words if w not in filler_words]

    # Final cleanup
    text = " ".join(words).strip()

    return text


def _parse_timestamps(text):
    """Parse VTT/SRT captions to get complete caption blocks with timing."""
    import re
    import webvtt
    import tempfile
    import requests

    captions = []

    # Extract the VTT URLs from the m3u8 playlist
    vtt_urls = []
    for line in text.splitlines():
        if line.startswith("http") and ".vtt" in line:
            vtt_urls.append(line.strip())

    # Download and parse each VTT file
    for vtt_url in vtt_urls:
        try:
            response = requests.get(vtt_url)
            vtt_content = response.text

            with tempfile.NamedTemporaryFile(suffix=".vtt", mode="w", delete=True) as f:
                f.write(vtt_content)
                f.flush()
                for caption in webvtt.read(f.name):
                    captions.append(
                        {
                            "text": caption.text.strip(),
                            "start": caption.start_in_seconds,
                            "end": caption.end_in_seconds,
                        }
                    )
        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not parse VTT file {vtt_url}: {str(e)}[/yellow]"
            )
            continue

    return captions


class VideoTranscriber:
    def __init__(self):
        self.console = Console()
        self.model = None
        self.temp_dir = tempfile.mkdtemp()
        self.captions = []
        self.transcribed_words = []

    def load_model(self):
        self.console.print(
            "[blue]Note: The following progress bar is from Whisper model download (tqdm). "
            "It will show percentage and speed, and will only appear the first time you run this script. "
            "Once the model is downloaded, the spinner and rich UI will resume as normal.[/blue]"
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            progress.add_task(description="Loading Whisper model...", total=None)
            self.model = WhisperModel("large-v3", device="auto", compute_type="auto")
        self.console.print("[green]✓ Whisper model loaded successfully[/green]")

    def extract_captions(self, url: str) -> List[Dict]:
        self.console.print("[yellow]Extracting captions...[/yellow]")
        ydl_opts = {
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en"],
            "skip_download": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if "subtitles" in info and "en" in info["subtitles"]:
                    subtitle_url = info["subtitles"]["en"][0]["url"]
                    self.console.print("[green]✓ Found English captions[/green]")
                    import requests

                    response = requests.get(subtitle_url)
                    caption_text = response.text
                    self.console.print(
                        "[bold blue]First 20 lines of captions file:[/bold blue]"
                    )
                    self.console.print("\n".join(caption_text.splitlines()[:20]))
                    word_timings = _parse_timestamps(caption_text)
                    self.console.print(
                        f"[bold blue]Extracted {len(word_timings)} word-level captions[/bold blue]"
                    )
                    return word_timings
                else:
                    self.console.print("[yellow]No English captions found[/yellow]")
                    return []
        except Exception as e:
            self.console.print(f"[red]Error extracting captions: {str(e)}[/red]")
            return []

    def download_audio(self, url: str) -> str:
        self.console.print(f"[yellow]Downloading audio from: {url}[/yellow]")
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=True,
        )
        download_task = None

        def hook(d):
            nonlocal download_task
            if d["status"] == "downloading":
                if download_task is None:
                    total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                    download_task = progress.add_task("Downloading", total=total)
                progress.update(download_task, completed=d.get("downloaded_bytes", 0))
            elif d["status"] == "finished":
                if download_task is not None:
                    progress.update(download_task, completed=progress.tasks[0].total)

        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": os.path.join(self.temp_dir, "audio.%(ext)s"),
            "progress_hooks": [hook],
            "noprogress": True,
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "force_generic_extractor": False,
            "nooverwrites": True,
            "extractor_args": {
                "brightcove": {
                    "player_id": "6057277730001",
                    "player_client": "default",
                    "player_key": "AQ~~,AAAAAAEL4GQ~,FdVP7EZCu5",
                    "account_id": "6057277730001",
                    "video_id": "6371886570112",
                }
            },
        }

        try:
            with progress:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
            audio_path = os.path.join(self.temp_dir, "audio.mp3")
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found at {audio_path}")
            self.console.print(
                f"[green]✓ Audio downloaded successfully to: {audio_path}[/green]"
            )
            return audio_path
        except Exception as e:
            self.console.print(f"[red]Error downloading audio: {str(e)}[/red]")
            raise

    def transcribe_audio(self, audio_path: str) -> List[Dict]:
        self.console.print(
            "[yellow]Transcribing audio with word timestamps...[/yellow]"
        )
        segments, _ = self.model.transcribe(
            audio_path, language="en", word_timestamps=True
        )
        words = []
        for segment in segments:
            if segment.words:
                for word in segment.words:
                    words.append(
                        {
                            "word": clean_text(word.word),
                            "start": word.start,
                            "end": word.end,
                        }
                    )
            else:
                words.append(
                    {
                        "word": clean_text(segment.text),
                        "start": segment.start,
                        "end": segment.end,
                    }
                )
        self.console.print(
            f"[green]✓ Transcribed {len(words)} words from audio[/green]"
        )
        return words

    def find_best_match(self, caption, transcribed_words, window=2.0):
        """Find the best matching transcribed text for a complete caption."""
        start, end = caption["start"], caption["end"]

        # Try different window sizes to find the best match
        window_sizes = [window, window * 1.5, window * 2.0]
        best_match = None
        best_accuracy = 0
        best_words = []

        original_text = clean_text(caption["text"])

        for current_window in window_sizes:
            # Allow a window before and after
            window_start = start - current_window
            window_end = end + current_window

            # Get words in the time window
            words_in_window = [
                w
                for w in transcribed_words
                if w["end"] >= window_start and w["start"] <= window_end
            ]

            if not words_in_window:
                continue

            # Join words to form the transcribed text for this window
            transcribed_text = " ".join(w["word"] for w in words_in_window)
            clean_transcribed = clean_text(transcribed_text)

            # Skip empty or whitespace-only texts
            if not clean_transcribed.strip() or not original_text.strip():
                continue

            # Calculate word error rate
            try:
                error = jiwer.process_words(original_text, clean_transcribed)
                total = error.hits + error.substitutions + error.deletions
                accuracy = (error.hits / total) * 100 if total > 0 else 0.0

                # Update best match if this is better
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_match = transcribed_text
                    best_words = words_in_window

            except Exception as e:
                console.print(
                    f"[yellow]Warning: Error comparing texts: {str(e)}[/yellow]"
                )
                continue

        return best_match or transcribed_text, best_words or words_in_window

    def process_video(self, url: str) -> List[Dict]:
        if not self.model:
            self.load_model()

        # Extract captions
        self.captions = self.extract_captions(url)
        self.console.print(
            f"[bold yellow]Extracted {len(self.captions)} captions[/bold yellow]"
        )

        # Download audio only
        audio_path = self.download_audio(url)

        # Transcribe the whole audio with word timestamps
        self.transcribed_words = self.transcribe_audio(audio_path)
        self.console.print(
            f"[bold yellow]Transcribed {len(self.transcribed_words)} words[/bold yellow]"
        )
        os.remove(audio_path)

        # --- NEW: Stitch all transcribed words into a single normalized string ---
        transcribed_full = " ".join([w["word"] for w in self.transcribed_words])
        transcribed_full_norm = clean_text(transcribed_full)
        transcribed_full_words = transcribed_full_norm.split()

        results = []
        for caption in self.captions:
            original_text = caption["text"]
            normalized_text = clean_text(original_text.lower())
            norm_words = normalized_text.split()
            n = len(norm_words)

            # --- Sliding window over the full transcription ---
            best_accuracy = 0.0
            best_window = ""
            best_error = None
            best_offset = None
            for i in range(len(transcribed_full_words) - n + 1):
                window_words = transcribed_full_words[i : i + n]
                window_text = " ".join(window_words)
                try:
                    error = jiwer.process_words(normalized_text, window_text)
                    total = error.hits + error.substitutions + error.deletions
                    accuracy = (error.hits / total) * 100 if total > 0 else 0.0
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_window = window_text
                        best_error = error
                        best_offset = i  # word offset in the full transcription
                except Exception as e:
                    continue

            # If no match found, skip
            if not best_window:
                continue

            # For reporting, try to estimate spoken_start/spoken_end from offset
            spoken_start = spoken_end = None
            if best_offset is not None and n > 0:
                # Map word offset to time using transcribed_words
                try:
                    spoken_start = self.transcribed_words[best_offset]["start"]
                    spoken_end = self.transcribed_words[best_offset + n - 1]["end"]
                except Exception:
                    pass

            # Status
            if best_accuracy >= 95:
                status = "PERFECT"
            elif best_accuracy >= 90:
                status = "GOOD"
            elif best_accuracy >= 80:
                status = "FAIR"
            else:
                status = "POOR"

            results.append(
                {
                    "caption_start": caption["start"],
                    "caption_end": caption["end"],
                    "original": original_text,
                    "normalized": normalized_text,
                    "transcribed": best_window,
                    "accuracy": best_accuracy,
                    "spoken_start": spoken_start,
                    "spoken_end": spoken_end,
                    "offset": (
                        spoken_start - caption["start"]
                        if spoken_start is not None
                        else None
                    ),
                    "status": status,
                    "errors": {
                        "substitutions": best_error.substitutions if best_error else 0,
                        "deletions": best_error.deletions if best_error else 0,
                        "insertions": best_error.insertions if best_error else 0,
                    },
                }
            )

        self.display_table(results)
        return results

    def display_table(self, results: List[Dict]):
        table = Table(title="Caption/Audio Sync Report", show_lines=True)
        table.add_column("Caption Start", style="cyan", justify="right")
        table.add_column("Caption End", style="cyan", justify="right")
        table.add_column("Original Caption", style="yellow")
        table.add_column("Normalized Original", style="blue")
        table.add_column("Transcribed (Whisper)", style="magenta")
        table.add_column("Accuracy %", style="green", justify="right")
        table.add_column("Offset (s)", style="red", justify="right")
        table.add_column("Status", style="bold", justify="center")
        table.add_column("Errors", style="blue", justify="right")

        for row in results:
            # Determine color based on accuracy
            if row["accuracy"] >= 95:
                acc_color = "green"
            elif row["accuracy"] >= 90:
                acc_color = "yellow"
            else:
                acc_color = "red"

            # Format errors
            errors = row.get("errors", {})
            error_str = f"S:{errors.get('substitutions', 0)} D:{errors.get('deletions', 0)} I:{errors.get('insertions', 0)}"

            # Format offset
            offset_str = f"{row['offset']:.2f}" if row["offset"] is not None else "-"

            # Determine status color
            status_color = {
                "PERFECT": "green",
                "GOOD": "yellow",
                "FAIR": "orange",
                "POOR": "red",
            }.get(row["status"], "white")

            # Highlight mismatches in transcribed text
            transcribed = row["transcribed"]
            if row["accuracy"] < 90:  # Only highlight if accuracy is less than 90%
                transcribed = f"[red]{transcribed}[/red]"

            table.add_row(
                f"{row['caption_start']:.2f}",
                f"{row['caption_end']:.2f}",
                row["original"],  # Show original exactly as extracted
                f"[blue]{row['normalized']}[/blue]",  # Show normalized version in blue
                transcribed,  # Show transcribed with potential highlighting
                f"[{acc_color}]{row['accuracy']:.1f}[/{acc_color}]",
                offset_str,
                f"[{status_color}]{row['status']}[/{status_color}]",
                error_str,
            )
        self.console.print(table)

    def save_mismatches(self, results: List[Dict], output_file: str = "matching.json"):
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        self.console.print(f"[green]✓ Results saved to {output_file}[/green]")


def main():
    transcriber = VideoTranscriber()
    url = "https://www.dell.com/support/contents/en-sg/videos/videoplayer/update-poweredge-drivers-using-a-dell-update-package-dup/6371886570112"
    try:
        results = transcriber.process_video(url)
        transcriber.save_mismatches(results)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
    finally:
        import shutil

        shutil.rmtree(transcriber.temp_dir)


if __name__ == "__main__":
    main()
