from __future__ import annotations

import csv
import tempfile
import warnings

import typer
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
import json
from enum import Enum
from pathlib import Path
import os
import gevent
import gevent.monkey
import shutil

gevent.monkey.patch_all(thread=False)

from transcribe.logging import LogLevel, initialize_logging  # noqa: E402
from transcribe.transcriber import Transcriber  # noqa: E402
from transcribe.processing import AudioExtractor  # noqa: E402
from transcribe.download import is_url, download_media  # noqa: E402

warnings.simplefilter(action="ignore", category=FutureWarning)


app = typer.Typer()
console = Console()
logger = initialize_logging(console)


class OutputFormat(str, Enum):
    CSV = "csv"
    SRT = "srt"
    VTT = "vtt"
    JSON = "json"
    TXT = "txt"


@app.command()
def transcribe(
    video_path: str,
    model_name: str = typer.Option("base", help="Whisper model to use"),
    output_format: OutputFormat = typer.Option(OutputFormat.CSV),
    output_path: str = typer.Option(None),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Verbosity level (-v: info, -vv: +debug, -vvv: +trace)",
    ),
    diarize: bool = typer.Option(
        False, "--diarize", "-d", help="Enable speaker diarization"
    ),
):
    """Transcribe a video file or URL with word-level timestamps and optional speaker diarization"""
    downloaded_file = None

    try:
        # Download if URL
        if is_url(video_path):
            downloaded_file = download_media(video_path)
            video_path = str(downloaded_file)

        # Initialize logging
        log_level = LogLevel(min(verbose + 2, LogLevel.DEBUG))
        logger = initialize_logging(console, log_level)

        # Handle output path
        if output_path:
            output_path = Path(output_path)
            if output_path.suffix[1:] not in OutputFormat:
                raise ValueError(f"Invalid output format: {output_path.suffix}")
            format = OutputFormat(output_path.suffix[1:])
        else:
            format = output_format
            output_path = get_output_filename(video_path, format)

        # Create progress bars with spinners
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        )

        # Set up progress tracking
        extract_task = progress.add_task("[cyan]Extracting audio...", total=None)
        transcribe_task = progress.add_task(
            "[cyan]Transcribing...", total=None, visible=False
        )
        if diarize:
            logger.warning("Speaker diarization is experimental and may be inaccurate.")
            diarize_task = progress.add_task(
                "[cyan]Identifying speakers...", total=1, visible=False
            )
        save_task = progress.add_task("[cyan]Saving...", total=1, visible=False)

        # Extract audio
        audio_path = "output_audio.wav"

        with progress:
            logger.info(f'Extracting audio from "{video_path}"')
            audio_path = extract_audio(video_path, progress, extract_task)

            # Transcribe
            progress.update(transcribe_task, visible=True)
            result = transcribe_audio(audio_path, progress, transcribe_task, model_name)
            logger.debug(f"Transcription result: {result}")

            # Perform diarization if requested
            if diarize:
                progress.update(diarize_task, visible=True)
                speakers = perform_diarization(audio_path, progress, diarize_task)
                result["segments"] = match_speakers_to_segments(
                    result["segments"], speakers
                )

            # Save results
            progress.update(save_task, visible=True)
            save_transcription(result, output_path, format)
            progress.update(save_task, advance=1)

        # Use the console directly for the completion message
        console.print(
            f' ✔️ Transcription completed and saved to [bold]"{output_path}"[/bold]'
        )

    finally:
        # Cleanup temporary audio file
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary audio file: {str(e)}")

        # Cleanup downloaded file
        if downloaded_file and downloaded_file.exists():
            downloaded_file.unlink()
            shutil.rmtree(downloaded_file.parent, ignore_errors=True)


def extract_audio(video_path: str, progress: Progress, task_id: int) -> Path:
    """Extract audio from video file using ffmpeg-python with progress monitoring."""
    with tempfile.NamedTemporaryFile("w+b", delete=False, suffix=".wav") as audio_file:
        extractor = AudioExtractor(logger)
        audio_path = audio_file.name
        extractor.extract_audio(video_path, audio_path, progress, task_id)
        return audio_path


def perform_diarization(audio_path: str, progress: Progress, task_id: int):
    """Perform speaker diarization."""
    from .diarizer import DiarizationManager

    diarizer = DiarizationManager(logger)
    return diarizer.perform_diarization(audio_path, progress, task_id)


def match_speakers_to_segments(segments: list, speakers: list):
    """Match transcribed segments with speaker labels."""
    for segment in segments:
        # Find the speaker who was talking during this segment
        segment_middle = (segment["start"] + segment["end"]) / 2
        for speaker in speakers:
            if speaker["start"] <= segment_middle <= speaker["end"]:
                segment["speaker"] = speaker["speaker"]
                break
        else:
            segment["speaker"] = "UNKNOWN"
    return segments


def transcribe_audio(
    audio_path: str,
    progress: Progress,
    task_id: int,
    model_name: str = "base",
):
    """Transcribe audio using OpenAI Whisper."""
    transcriber = Transcriber(model_name=model_name)
    return transcriber.transcribe(audio_path, progress, task_id)


def get_output_filename(
    video_path: str, format: OutputFormat = OutputFormat.CSV
) -> Path:
    """Generate output filename in current working directory"""
    video_name = Path(video_path).stem
    return Path.cwd() / f"{video_name}.{format.value}"


def save_transcription(result: dict, output_path: Path, format: OutputFormat):
    """Save transcription in specified format."""
    try:
        logger.info(f'Saving transcription to "{output_path}"')
        with open(output_path, "w", encoding="utf-8") as f:
            if format == OutputFormat.CSV:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, escapechar="\\")
                writer.writerow(["start", "end", "speaker", "text"])
                for segment in result["segments"]:
                    for word in segment.get("words", []):
                        writer.writerow(
                            [
                                f"{word['start']:.2f}",
                                f"{word['end']:.2f}",
                                segment.get("speaker", "UNKNOWN"),
                                word["word"].strip(),
                            ]
                        )
            elif format == OutputFormat.SRT:
                for i, segment in enumerate(result["segments"], 1):
                    speaker = segment.get("speaker", "UNKNOWN")
                    f.write(f"{i}\n")
                    f.write(
                        f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
                    )
                    f.write(f"[{speaker}] {segment['text']}\n\n")
            elif format == OutputFormat.VTT:
                f.write("WEBVTT\n\n")
                for segment in result["segments"]:
                    speaker = segment.get("speaker", "UNKNOWN")
                    f.write(
                        f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
                    )
                    f.write(f"[{speaker}] {segment['text']}\n\n")
            elif format == OutputFormat.JSON:
                json.dump(result, f, indent=2)
            else:  # TXT
                for segment in result["segments"]:
                    speaker = segment.get("speaker", "UNKNOWN")
                    f.write(f"[{speaker}] {segment['text']}\n")
    except Exception as e:
        logger.error(f"Error saving transcription: {str(e)}")
        raise typer.Exit(1)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT/VTT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")


if __name__ == "__main__":
    app()
