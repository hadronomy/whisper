from typing import List, Tuple, Optional, Iterable
from dataclasses import dataclass
import faster_whisper
import logging
import numpy as np

logger = logging.getLogger("transcribe")


@dataclass
class TranscriptionInfo:
    language: str
    language_probability: float
    duration: float
    duration_after_vad: float
    all_language_probs: Optional[List[Tuple[str, float]]] = None


class WhisperPipeline:
    def __init__(self, model: str, device: str, batch_size: int = 8):
        self.compute_types = {"cpu": "int8", "cuda": "float16"}
        self.model = faster_whisper.WhisperModel(
            model, device=device, compute_type=self.compute_types[device]
        )
        self.pipeline = (
            faster_whisper.BatchedInferencePipeline(self.model)
            if batch_size > 0
            else None
        )
        self.batch_size = batch_size
        self._progress_callback = None
        self._total_segments = 0
        self._processed_segments = 0

    def set_progress_callback(self, callback: Optional[callable]):
        """Set a callback function to report progress"""
        self._progress_callback = callback

    def _report_progress(self, amount: int = 1):
        """Internal method to report progress"""
        self._processed_segments += amount
        if self._progress_callback:
            self._progress_callback(self._processed_segments, self._total_segments)

    def transcribe(
        self,
        audio: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        word_timestamps: bool = True,
        vad_filter: bool = True,
    ) -> Tuple[Iterable[faster_whisper.transcribe.Segment], TranscriptionInfo]:
        """
        Transcribe audio with progress tracking.

        Args:
            audio: Path to audio file or audio data
            language: Optional language code
            task: Transcription task ("transcribe" or "translate")
            word_timestamps: Whether to include word-level timestamps
            vad_filter: Whether to apply voice activity detection

        Returns:
            Tuple of (segments, transcription info)
        """
        try:
            # Reset progress tracking
            self._processed_segments = 0

            # Decode audio and get segments
            if not isinstance(audio, np.ndarray):
                audio_data = faster_whisper.decode_audio(audio)
            else:
                audio_data = audio

            # Estimate total segments based on audio length
            audio_duration = (
                len(audio_data) / self.model.feature_extractor.sampling_rate
            )
            self._total_segments = max(
                1, int(audio_duration / 30)
            )  # Rough estimate: ~30 sec per segment

            # Use batched pipeline if batch_size > 0, otherwise use regular model
            if self.batch_size > 0:
                segments, info = self.pipeline.transcribe(
                    audio_data,
                    language=language,
                    batch_size=self.batch_size,
                    task=task,
                    word_timestamps=word_timestamps,
                    vad_filter=vad_filter,
                )
            else:
                segments, info = self.model.transcribe(
                    audio_data,
                    language=language,
                    task=task,
                    word_timestamps=word_timestamps,
                    vad_filter=vad_filter,
                )

            # Convert to list if we need progress tracking
            if self._progress_callback:
                segments = list(segments)
                # Update total count with actual number
                self._total_segments = len(segments)
                # Process segments with progress tracking
                processed_segments = []
                for segment in segments:
                    processed_segments.append(segment)
                    self._report_progress()
                segments = processed_segments

            return segments, TranscriptionInfo(
                language=info.language,
                language_probability=info.language_probability,
                duration=info.duration,
                duration_after_vad=info.duration_after_vad,
                all_language_probs=info.all_language_probs,
            )

        except Exception as e:
            logger.error(f"Pipeline transcription error: {str(e)}")
            raise

    def encode(self, features):
        """Encode features using the model"""
        return self.model.encode(features)

    def decode(self, encoded_features, **kwargs):
        """Decode encoded features using the model"""
        return self.model.decode(encoded_features, **kwargs)

    def generate_with_fallback(self, *args, **kwargs):
        """Generate transcription with fallback options"""
        return self.model.generate_with_fallback(*args, **kwargs)

    def align(self, *args, **kwargs):
        """Align transcription with audio"""
        return self.model.align(*args, **kwargs)

    @property
    def is_multilingual(self):
        """Check if the model supports multiple languages"""
        return self.model.is_multilingual

    @property
    def feature_extractor(self):
        """Get the feature extractor instance"""
        return self.model.feature_extractor
