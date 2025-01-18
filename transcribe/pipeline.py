import faster_whisper
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class TranscriptionInfo:
    language: str
    language_probability: float
    duration: float
    duration_after_vad: float
    all_language_probs: Optional[List[Tuple[str, float]]] = None

class WhisperPipeline:
    def __init__(self, model_name: str, device: str, batch_size: int = 8):
        self.compute_types = {"cpu": "int8", "cuda": "float16"}
        self.model = faster_whisper.WhisperModel(
            model_name,
            device=device,
            compute_type=self.compute_types[device]
        )
        self.pipeline = faster_whisper.BatchedInferencePipeline(self.model)
        self.batch_size = batch_size

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Tuple[List[faster_whisper.transcribe.Segment], TranscriptionInfo]:
        audio = faster_whisper.decode_audio(audio_path)
        
        segments, info = (
            self.pipeline.transcribe(
                audio,
                language=language,
                batch_size=self.batch_size,
                task=task,
                word_timestamps=True,  # Enable word timestamps
                vad_filter=True,       # Enable voice activity detection
            ) if self.batch_size > 0
            else self.model.transcribe(
                audio,
                language=language,
                word_timestamps=True,  # Enable word timestamps
                vad_filter=True       # Enable voice activity detection
            )
        )
        
        return segments, TranscriptionInfo(
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            duration_after_vad=info.duration_after_vad,
            all_language_probs=info.all_language_probs
        )