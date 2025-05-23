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
    # Convert to lowercase
    text = text.lower()

    # Replace domain names with generic placeholder
    text = re.sub(r"[a-z0-9]+\.com", "website", text)
    text = re.sub(r"[a-z0-9]+\.org", "website", text)
    text = re.sub(r"[a-z0-9]+\.net", "website", text)

    # Remove special characters except apostrophes
    text = "".join(c for c in text if c.isalnum() or c == "'")

    # Remove extra whitespace
    text = " ".join(text.split())

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
        # Allow a window before and after
        window_start = start - window
        window_end = end + window

        # Get words in the time window
        words_in_window = [
            w
            for w in transcribed_words
            if w["end"] >= window_start and w["start"] <= window_end
        ]

        if not words_in_window:
            return "", []

        # Join words to form the transcribed text for this window
        transcribed_text = " ".join(w["word"] for w in words_in_window)
        original_text = caption["text"]

        # Clean both texts for comparison
        clean_transcribed = clean_text(transcribed_text)
        clean_original = clean_text(original_text)

        # Skip empty or whitespace-only texts
        if not clean_transcribed.strip() or not clean_original.strip():
            return transcribed_text, words_in_window

        # Calculate word error rate
        try:
            error = jiwer.process_words(clean_original, clean_transcribed)
            total_errors = error.substitutions + error.deletions + error.insertions
            total_words = error.hits + error.substitutions + error.deletions

            # If there are significant errors, try to find a better match by adjusting the window
            if total_errors > 0 and total_words > 0:
                # Try with a larger window
                larger_window = window * 1.5
                larger_words = [
                    w
                    for w in transcribed_words
                    if w["end"] >= start - larger_window
                    and w["start"] <= end + larger_window
                ]
                if larger_words:
                    larger_text = " ".join(w["word"] for w in larger_words)
                    larger_clean = clean_text(larger_text)
                    larger_error = jiwer.process_words(clean_original, larger_clean)
                    larger_total_errors = (
                        larger_error.substitutions
                        + larger_error.deletions
                        + larger_error.insertions
                    )

                    # Use the larger window if it gives better results
                    if larger_total_errors < total_errors:
                        return larger_text, larger_words

            return transcribed_text, words_in_window

        except Exception as e:
            console.print(f"[yellow]Warning: Error comparing texts: {str(e)}[/yellow]")
            console.print(f"[yellow]Original: {clean_original}[/yellow]")
            console.print(f"[yellow]Transcribed: {clean_transcribed}[/yellow]")
            return transcribed_text, words_in_window

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

        # For each caption, find the best matching transcribed text
        results = []
        for caption in self.captions:
            original = clean_text(caption["text"])
            transcribed, words_in_window = self.find_best_match(
                caption, self.transcribed_words, window=2.0
            )

            # Skip empty or whitespace-only texts
            if not original.strip() or not transcribed.strip():
                continue

            self.console.print(
                f"[cyan]Caption: {caption['text']} | Start: {caption['start']} | End: {caption['end']} | Words in window: {len(words_in_window)}[/cyan]"
            )

            # Calculate accuracy (WER)
            try:
                error = jiwer.process_words(original, transcribed)
                total = error.hits + error.substitutions + error.deletions
                accuracy = (error.hits / total) * 100 if total > 0 else 0.0

                # Find actual spoken start/end if any words matched
                if words_in_window:
                    spoken_start = words_in_window[0]["start"]
                    spoken_end = words_in_window[-1]["end"]
                    offset = spoken_start - caption["start"]
                else:
                    spoken_start = spoken_end = offset = None

                # Determine match/mismatch status with more granular thresholds
                if accuracy >= 95:
                    status = "PERFECT"
                elif accuracy >= 90:
                    status = "GOOD"
                elif accuracy >= 80:
                    status = "FAIR"
                else:
                    status = "POOR"

                results.append(
                    {
                        "caption_start": caption["start"],
                        "caption_end": caption["end"],
                        "original": caption["text"],
                        "transcribed": transcribed,
                        "accuracy": accuracy,
                        "spoken_start": spoken_start,
                        "spoken_end": spoken_end,
                        "offset": offset,
                        "status": status,
                        "errors": {
                            "substitutions": error.substitutions,
                            "deletions": error.deletions,
                            "insertions": error.insertions,
                        },
                    }
                )
            except Exception as e:
                self.console.print(
                    f"[yellow]Warning: Error processing caption: {str(e)}[/yellow]"
                )
                continue

        self.display_table(results)
        return results

    def display_table(self, results: List[Dict]):
        table = Table(title="Caption/Audio Sync Report", show_lines=True)
        table.add_column("Caption Start", style="cyan", justify="right")
        table.add_column("Caption End", style="cyan", justify="right")
        table.add_column("Original Caption", style="yellow")
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

            table.add_row(
                f"{row['caption_start']:.2f}",
                f"{row['caption_end']:.2f}",
                row["original"],
                row["transcribed"],
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
