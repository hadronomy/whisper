import logging
from typing import Optional, Dict, Any
from rich.progress import Progress
import faster_whisper
from tqdm import tqdm as tqdm

from transcribe.device import DeviceManager
from transcribe.pipeline import WhisperPipeline

logger = logging.getLogger("transcribe")


class RichTqdm:
    """A tqdm-like class that forwards updates to Rich progress bars"""

    def __init__(self, progress: Progress, task_id: int, **kwargs):
        self.progress = progress
        self.task_id = task_id
        total = kwargs.get("total")
        if total:
            self.progress.update(self.task_id, total=total)

    def update(self, n=1):
        self.progress.update(self.task_id, advance=n)

    def close(self):
        pass


class Transcriber:
    def __init__(
        self,
        model: str = "base",
        device: Optional[str] = None,
        batch_size: int = 8,
        default_language: Optional[str] = None,
    ):
        """Initialize transcriber with specified configuration."""
        self.model = model
        self.device = device or DeviceManager.get_default_device()
        self.batch_size = batch_size
        self.default_language = default_language
        self._pipeline = None

    @property
    def pipeline(self) -> WhisperPipeline:
        """Lazy load the whisper pipeline."""
        if self._pipeline is None:
            self._pipeline = WhisperPipeline(self.model, self.device, self.batch_size)
        return self._pipeline

    def transcribe(
        self,
        audio_path: str,
        progress: Optional[Progress] = None,
        task_id: Optional[int] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper with progress tracking.

        Args:
            audio_path: Path to the audio file
            progress: Optional Progress instance for tracking
            task_id: Optional task ID for progress tracking
            language: Optional language code to force transcription language

        Returns:
            Dictionary containing transcription results
        """
        try:
            # Override tqdm if we have progress tracking
            original_tqdm = None
            if progress and task_id is not None:
                original_tqdm = faster_whisper.transcribe.tqdm
                # Replace tqdm with our Rich-compatible version
                faster_whisper.transcribe.tqdm = lambda *args, **kwargs: RichTqdm(
                    progress, task_id, **kwargs
                )

            try:
                segments, info = self.pipeline.transcribe(
                    audio_path,
                    language=language or self.default_language,
                    vad_filter=True,
                )

                segment_list = list(segments)

                return {
                    "text": "".join(
                        getattr(segment, "text", "") for segment in segment_list
                    ),
                    "segments": [
                        self._process_segment(segment, i)
                        for i, segment in enumerate(segment_list)
                    ],
                    "language": getattr(info, "language", None),
                    "language_probability": getattr(info, "language_probability", 0.0),
                    "audio_path": audio_path,
                }
            finally:
                # Restore original tqdm
                if original_tqdm:
                    faster_whisper.transcribe.tqdm = original_tqdm

        except Exception as e:
            error_msg = f"Transcription error for {audio_path}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _process_segment(self, segment: Any, index: int) -> Dict[str, Any]:
        """Process a single segment with error handling."""
        try:
            return {
                "id": index,
                "seek": getattr(segment, "seek", 0),
                "start": getattr(segment, "start", 0),
                "end": getattr(segment, "end", 0),
                "text": getattr(segment, "text", ""),
                "tokens": getattr(segment, "tokens", []),
                "temperature": getattr(segment, "temperature", 0.0),
                "avg_logprob": getattr(segment, "avg_logprob", 0.0),
                "compression_ratio": getattr(segment, "compression_ratio", 0.0),
                "no_speech_prob": getattr(segment, "no_speech_prob", 0.0),
                "words": [
                    {
                        "word": getattr(word, "word", ""),
                        "start": getattr(word, "start", 0),
                        "end": getattr(word, "end", 0),
                        "probability": getattr(word, "probability", 0.0),
                    }
                    for word in (getattr(segment, "words", []) or [])
                ],
            }
        except Exception as e:
            logger.warning(f"Error processing segment {index}: {str(e)}")
            return {
                "id": index,
                "text": f"[Error processing segment {index}]",
                "words": [],
            }
