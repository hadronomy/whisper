import contextlib
import logging
import shutil
import tempfile
import ffmpeg
import socket
import os
import gevent
from rich.progress import Progress

logger = logging.getLogger("transcribe")


class AudioExtractor:
    def __init__(self, logger=logger):
        self.tmpdir = None
        self.sock_path = None
        self.sock = None
        self.logger = logger

    def setup_socket(self) -> None:
        """Setup Unix socket for FFmpeg progress monitoring."""
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock_path = os.path.join(self.tmpdir, "progress")
        self.sock.bind(self.sock_path)
        self.sock.listen(1)

    def get_duration(self, video_path: str) -> float:
        """Get duration of video file."""
        probe = ffmpeg.probe(video_path)
        return float(probe["streams"][0]["duration"])

    def extract_audio(
        self, video_path: str, audio_path: str, progress: Progress, task_id: int
    ) -> bool:
        """Extract audio from video file using ffmpeg-python."""
        try:
            duration = self.get_duration(video_path)
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
            self.logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleanup socket and temporary files."""
        if self.sock:
            self.sock.close()
        if self.sock_path and os.path.exists(self.sock_path):
            os.remove(self.sock_path)


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
def _tmpdir_scope():
    tmpdir = tempfile.mkdtemp()
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir)
