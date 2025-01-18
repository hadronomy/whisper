import logging
import warnings

import torch
import whisper
from rich.progress import Progress

from transcribe.device import DeviceManager

logger = logging.getLogger("transcribe")


class Transcriber:
    def __init__(self, model_name: str = "base", device: str = None):
        """Initialize transcriber with specified model and device."""
        self.model_name = model_name
        self.device = device or DeviceManager.get_default_device()
        self.model = None

    def load_model(self):
        """Load the Whisper model."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            torch.set_default_device(self.device)
            self.model = whisper.load_model(self.model_name, self.device)

    def transcribe(
        self,
        audio_path: str,
        progress: Progress = None,
        task_id: int = None,
    ) -> dict:
        """Transcribe audio using OpenAI Whisper."""
        try:
            if not self.model:
                self.load_model()

            result = self.model.transcribe(audio_path, word_timestamps=True)

            if progress and task_id is not None:
                progress.update(task_id, advance=1)

            return result

        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise RuntimeError(f"Transcription error: {str(e)}")
