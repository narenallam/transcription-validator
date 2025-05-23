# Video Transcription and Caption Comparison Tool

This Python application downloads videos from various sources (YouTube, Vimeo, etc.), extracts audio, transcribes it using OpenAI's Whisper large-v3 model, and compares the transcription with captions.

## Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system
- CUDA-compatible GPU (recommended for faster processing)

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the script:
```bash
python video_transcriber.py
```

2. Enter the video URL when prompted.

3. The script will:
   - Download the video
   - Extract audio
   - Split it into 15-second chunks
   - Transcribe each chunk using Whisper
   - Compare with captions
   - Generate a mismatches.json file

## Output

The script generates a `mismatches.json` file with the following structure:
```json
{
  "start": "1.50",
  "end": "3.02",
  "original": "to update drivers on your poweredge server",
  "transcribed": "update drivers on your poweredge"
}
```

## Notes

- The first run will download the Whisper large-v3 model (~2.9GB)
- Processing time depends on your hardware and video length
- Temporary files are automatically cleaned up after processing 