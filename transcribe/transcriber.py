import logging

from rich.progress import Progress

from transcribe.device import DeviceManager
from transcribe.pipeline import WhisperPipeline

logger = logging.getLogger("transcribe")


class Transcriber:
    def __init__(
        self, model_name: str = "base", device: str = None, batch_size: int = 8
    ):
        """Initialize transcriber with specified model and device."""
        self.model_name = model_name
        self.device = device or DeviceManager.get_default_device()
        self.batch_size = batch_size
        self._pipeline = None

    @property
    def pipeline(self) -> WhisperPipeline:
        """Lazy load the whisper pipeline."""
        if self._pipeline is None:
            self._pipeline = WhisperPipeline(
                self.model_name, self.device, self.batch_size
            )
        return self._pipeline

    def transcribe(
        self,
        audio_path: str,
        progress: Progress = None,
        task_id: int = None,
        language: str = None,
    ) -> dict:
        """Transcribe audio using Whisper."""
        try:
            segments, info = self.pipeline.transcribe(audio_path, language=language)

            segment_list = list(segments)

            # Structure result with segments
            result = {
                "text": "".join(segment.text for segment in segment_list),
                "segments": [
                    {
                        "id": i,
                        "seek": segment.seek,
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                        "tokens": segment.tokens,
                        "temperature": segment.temperature,
                        "avg_logprob": segment.avg_logprob,
                        "compression_ratio": segment.compression_ratio,
                        "no_speech_prob": segment.no_speech_prob,
                        "words": [
                            {
                                "word": word.word,
                                "start": word.start,
                                "end": word.end,
                                "probability": word.probability,
                            }
                            for word in (segment.words or [])
                        ],
                    }
                    for i, segment in enumerate(segment_list)
                ],
                "language": info.language,
                "language_probability": info.language_probability,
            }

            if progress and task_id is not None:
                progress.update(task_id, advance=1)

            return result

        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise RuntimeError(f"Transcription error: {str(e)}")
