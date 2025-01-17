from __future__ import annotations
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
import whisper
import ffmpeg
import json
from enum import Enum
from pathlib import Path
import contextlib
from contextlib import contextmanager
import tempfile
import shutil
import socket
import os
import gevent
import gevent.monkey
import torch
import logging

gevent.monkey.patch_all(thread=False)

warnings.simplefilter(action="ignore", category=FutureWarning)


@contextmanager
def device_context(device=None):
    """Context manager to ensure consistent device usage."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        yield device
    finally:
        torch.cuda.empty_cache() if device == "cuda" else None


class DeviceManager:
    @staticmethod
    def get_default_device():
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def ensure_same_device(tensor, target_device):
        return tensor.to(target_device) if tensor.device != target_device else tensor


# Set up custom logger that uses rich console
class RichLogger:
    def __init__(self, console: Console):
        self.console = console
        self.device = DeviceManager.get_default_device()

    def error(self, msg):
        self.console.print(f"[red]ERROR [{self.device}]: {msg}[/red]")

    def warning(self, msg):
        self.console.print(f"[yellow]WARNING [{self.device}]: {msg}[/yellow]")

    def info(self, msg):
        self.console.print(f"[blue]INFO [{self.device}]: {msg}[/blue]")

    def debug(self, msg):
        self.console.print(f"[dim]DEBUG [{self.device}]: {msg}[/dim]")


app = typer.Typer()
console = Console()
logger = RichLogger(console)

logging.getLogger("speechbrain").addHandler(logging.NullHandler())
logging.getLogger("whisper").addHandler(logging.NullHandler())
logging.getLogger("ffmpeg").addHandler(logging.NullHandler())


class OutputFormat(str, Enum):
    CSV = "csv"
    SRT = "srt"
    VTT = "vtt"
    JSON = "json"
    TXT = "txt"


@contextlib.contextmanager
def _tmpdir_scope():
    tmpdir = tempfile.mkdtemp()
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir)


def _do_watch_progress(filename, sock, progress, task_id):
    """Watch FFmpeg progress events from unix socket."""
    connection, client_address = sock.accept()
    data = b""
    try:
        while True:
            more_data = connection.recv(16)
            if not more_data:
                break
            data += more_data
            lines = data.split(b"\n")
            for line in lines[:-1]:
                line = line.decode()
                parts = line.split("=")
                key = parts[0] if len(parts) > 0 else None
                value = parts[1] if len(parts) > 1 else None

                if key == "out_time_ms":
                    time = round(float(value) / 1000000.0, 2)
                    progress.update(task_id, completed=int(time))
                elif key == "progress" and value == "end":
                    progress.update(task_id, completed=progress.tasks[task_id].total)
            data = lines[-1]
    finally:
        connection.close()


@contextlib.contextmanager
def _watch_progress(progress, task_id):
    """Set up unix socket for FFmpeg progress monitoring."""
    with _tmpdir_scope() as tmpdir:
        socket_filename = os.path.join(tmpdir, "sock")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        with contextlib.closing(sock):
            sock.bind(socket_filename)
            sock.listen(1)
            child = gevent.spawn(
                _do_watch_progress, socket_filename, sock, progress, task_id
            )
            try:
                yield socket_filename
            except:
                gevent.kill(child)
                raise


def extract_audio(video_path: str, audio_path: str, progress: Progress, task_id: int):
    """Extract audio from video file using ffmpeg-python with progress monitoring."""
    try:
        probe = ffmpeg.probe(video_path)
        duration = float(probe["streams"][0]["duration"])
        progress.update(task_id, total=int(duration))

        with _watch_progress(progress, task_id) as socket_filename:
            (
                ffmpeg.input(video_path)
                .output(audio_path, ac=1, ar=16000)
                .global_args("-progress", f"unix://{socket_filename}")
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        return True
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error: {e.stderr.decode()}")
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}")


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
    console: Console,
    model_name: str = "base",
    verbose: bool = False,
):
    """Transcribe audio using OpenAI Whisper."""
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            # Ensure device consistency with diarization
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch.set_default_device(device)
            model = whisper.load_model(model_name, device)

        # Transcribe without progress callback
        result = model.transcribe(audio_path, word_timestamps=True)

        # Update progress after completion
        progress.update(task_id, advance=1)

        # Print transcription if verbose mode is enabled
        if verbose:
            for segment in result["segments"]:
                logger.info(f"Transcript: {segment['text']}")

        return result
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise RuntimeError(f"Transcription error: {str(e)}")


def get_output_filename(
    video_path: str, format: OutputFormat = OutputFormat.CSV
) -> Path:
    """Generate output filename in current working directory"""
    video_name = Path(video_path).stem
    return Path.cwd() / f"{video_name}.{format.value}"


def save_transcription(result: dict, output_path: Path, format: OutputFormat):
    """Save transcription in specified format."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            if format == OutputFormat.CSV:
                f.write("start,end,speaker,text\n")
                for segment in result["segments"]:
                    for word in segment.get("words", []):
                        speaker = segment.get("speaker", "UNKNOWN")
                        f.write(
                            f"{word['start']:.2f},{word['end']:.2f},{speaker},{word['word']}\n"
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


@app.command()
def transcribe(
    video_path: str,
    model_name: str = typer.Option("base", help="Whisper model to use"),
    output_format: OutputFormat = typer.Option(OutputFormat.CSV),
    output_path: str = typer.Option(None),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show transcription output"
    ),
    diarize: bool = typer.Option(
        False, "--diarize", "-d", help="Enable speaker diarization"
    ),
):
    """Transcribe a video file with word-level timestamps and optional speaker diarization"""

    try:
        # Handle output path
        if output_path:
            output_path = Path(output_path)
            if output_path.suffix[1:] not in OutputFormat._member_names_:
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
            "[cyan]Transcribing...", total=1, visible=False
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
            extract_audio(video_path, audio_path, progress, extract_task)

            # Transcribe
            progress.update(transcribe_task, visible=True)
            result = transcribe_audio(
                audio_path, progress, transcribe_task, console, model_name, verbose
            )

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
            f"[green]Transcription completed and saved to {output_path}[/green]"
        )

    finally:
        # Cleanup temporary audio file
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary audio file: {str(e)}")


if __name__ == "__main__":
    app()
