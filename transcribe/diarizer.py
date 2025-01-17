from pyannote.audio import Pipeline
import torch
import torchaudio
import os
import tempfile
import contextlib
import warnings
import logging
from rich.progress import Progress


class DiarizationManager:
    """Manages speaker diarization process using Pyannote.audio pipeline."""

    def __init__(self, logger):
        self.logger = logger
        self._setup_logging()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize Pyannote pipeline with pretrained model
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading {os.getenv('HF_TOKEN')}")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0", use_auth_token=os.getenv("HF_TOKEN")
        ).to(self.device)

    def _setup_logging(self):
        """Configure logging for related libraries."""
        libraries = ["pyannote", "torch", "torchaudio"]
        for lib in libraries:
            lib_logger = logging.getLogger(lib)
            lib_logger.setLevel(logging.ERROR)
            for handler in lib_logger.handlers[:]:
                lib_logger.removeHandler(handler)
            lib_logger.addHandler(logging.NullHandler())
            lib_logger.propagate = False

    @contextlib.contextmanager
    def _prepare_audio(self, audio_path: str):
        """Prepare audio file for diarization by converting to mono."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Load and resample audio to 16kHz mono
                waveform, sample_rate = torchaudio.load(audio_path)
                if waveform.shape[0] > 1:  # If not mono
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)

                mono_path = os.path.join(temp_dir, "mono_file.wav")
                torchaudio.save(mono_path, waveform, 16000)

                yield temp_dir, mono_path
            except Exception as e:
                self.logger.error(f"Error preparing audio: {str(e)}")
                raise e

    @contextlib.contextmanager
    def _suppress_warnings(self):
        """Context manager to suppress warnings during diarization."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            yield

    def perform_diarization(
        self, audio_path: str, progress: Progress, task_id: int
    ) -> list:
        """Perform speaker diarization using Pyannote."""
        try:
            with self._suppress_warnings():
                self.logger.info(f"Starting diarization using {self.device}")

                with self._prepare_audio(audio_path) as (temp_dir, mono_path):
                    self.logger.info("Running diarization...")

                    # Run diarization
                    diarization = self.pipeline(mono_path)

                    # Convert results to our format
                    speakers = []
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        speakers.append(
                            {
                                "start": turn.start,
                                "end": turn.end,
                                "speaker": f"SPEAKER_{speaker.split('#')[-1]}",
                            }
                        )

                    # Clean up CUDA memory
                    torch.cuda.empty_cache()

                    self.logger.info(
                        f"Diarization completed. Found {len(speakers)} speaker segments."
                    )
                    progress.update(task_id, advance=1)

                    return speakers

        except Exception as e:
            self.logger.error(f"Diarization failed: {str(e)}")
            raise RuntimeError(f"Diarization error: {str(e)}")
