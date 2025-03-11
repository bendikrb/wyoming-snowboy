#!/usr/bin/env python3
import argparse
import asyncio
import itertools
import logging
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, Final, List, Optional, Set

from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, WakeModel, WakeProgram
from wyoming.server import AsyncEventHandler, AsyncServer, AsyncTcpServer
from wyoming.wake import Detect, Detection, NotDetected

from . import __version__, snowboydetect

_LOGGER = logging.getLogger()
_DIR = Path(__file__).parent

SAMPLES_PER_CHUNK: Final = 1024
BYTES_PER_CHUNK: Final = SAMPLES_PER_CHUNK * 2  # 16-bit
DEFAULT_KEYWORD: Final = "snowboy"


@dataclass
class KeywordSettings:
    sensitivity: Optional[float] = None
    audio_gain: Optional[float] = None
    apply_frontend: Optional[bool] = None
    num_keywords: int = 1


# https://kalliope-project.github.io/kalliope/settings/triggers/snowboy/
DEFAULT_SETTINGS: Dict[str, KeywordSettings] = {
    "alexa": KeywordSettings(apply_frontend=True),
    "snowboy": KeywordSettings(apply_frontend=False),
    "jarvis": KeywordSettings(num_keywords=2, apply_frontend=True),
    "smart_mirror": KeywordSettings(apply_frontend=False),
    "subex": KeywordSettings(apply_frontend=True),
    "neoya": KeywordSettings(num_keywords=2, apply_frontend=True),
    "computer": KeywordSettings(apply_frontend=True),
    "view_glass": KeywordSettings(apply_frontend=True),
}


@dataclass
class Keyword:
    """Single snowboy keyword"""

    name: str
    model_path: Path
    settings: KeywordSettings


@dataclass
class KeywordDetector:
    """Keyword detector with state"""

    name: str
    detector: snowboydetect.SnowboyDetect
    is_detected: bool = False


@dataclass
class ClientData:
    """Data for a connected client"""

    active_keywords: Optional[Set[str]] = field(default_factory=set)
    detectors: Optional[Dict[str, KeywordDetector]] = field(default_factory=dict)
    audio_buffer: Optional[bytes] = None


class State:
    """State of system"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.available_keywords: Dict[str, Keyword] = {}
        self.clients: Dict[str, ClientData] = {}
        self._load_available_keywords()

    def _load_available_keywords(self):
        """Load all available keywords from data and custom directories"""
        for kw_dir in self.args.custom_model_dir + [self.args.data_dir]:
            if not kw_dir.is_dir():
                continue

            for kw_path in itertools.chain(
                kw_dir.glob("*.umdl"), kw_dir.glob("*.pmdl")
            ):
                kw_name = kw_path.stem
                if kw_name not in self.available_keywords:
                    self.available_keywords[kw_name] = Keyword(
                        name=kw_name,
                        model_path=kw_path,
                        settings=DEFAULT_SETTINGS.get(kw_name, KeywordSettings()),
                    )
        _LOGGER.debug(
            f"Loaded {len(self.available_keywords)} available keywords: {', '.join(self.available_keywords.keys())}"
        )

    def create_detector(self, keyword_name: str) -> KeywordDetector:
        """Create a detector for a specific keyword"""
        if keyword_name not in self.available_keywords:
            raise ValueError(f"No keyword {keyword_name}")

        keyword = self.available_keywords[keyword_name]

        sensitivity = self.args.sensitivity
        if keyword.settings.sensitivity is not None:
            sensitivity = keyword.settings.sensitivity

        sensitivity_str = ",".join(
            str(sensitivity) for _ in range(keyword.settings.num_keywords)
        )

        audio_gain = self.args.audio_gain
        if keyword.settings.audio_gain is not None:
            audio_gain = keyword.settings.audio_gain

        apply_frontend = self.args.apply_frontend
        if keyword.settings.apply_frontend is not None:
            apply_frontend = keyword.settings.apply_frontend

        _LOGGER.debug(
            "Loading %s with sensitivity=%s, audio_gain=%s, apply_frontend=%s",
            keyword.name,
            sensitivity_str,
            audio_gain,
            apply_frontend,
        )

        detector = snowboydetect.SnowboyDetect(
            str(self.args.data_dir / "common.res").encode(),
            str(keyword.model_path).encode(),
        )

        detector.SetSensitivity(sensitivity_str.encode())
        detector.SetAudioGain(audio_gain)
        detector.ApplyFrontend(apply_frontend)

        return KeywordDetector(name=keyword_name, detector=detector)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="stdio://", help="unix:// or tcp://")
    parser.add_argument(
        "--data-dir",
        default=_DIR / "data",
        help="Path to directory with default keywords",
    )
    parser.add_argument(
        "--custom-model-dir",
        action="append",
        default=[],
        help="Path to directory with custom wake word models (*.pmdl, *.umdl)",
    )
    #
    parser.add_argument("--sensitivity", type=float, default=0.5)
    parser.add_argument("--audio-gain", type=float, default=1.0)
    parser.add_argument("--apply-frontend", action="store_true")
    #
    parser.add_argument(
        "--zeroconf",
        nargs="?",
        const="snowboy",
        help="Enable discovery over zeroconf with optional name (default: snowboy)",
    )
    #
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    if args.version:
        print(__version__)
        return

    args.data_dir = Path(args.data_dir)
    args.custom_model_dir = [Path(p) for p in args.custom_model_dir]

    state = State(args=args)

    _LOGGER.info("Ready")

    # Start server
    server = AsyncServer.from_uri(args.uri)

    if args.zeroconf:
        if not isinstance(server, AsyncTcpServer):
            raise ValueError("Zeroconf requires tcp:// uri")

        from wyoming.zeroconf import register_server

        tcp_server: AsyncTcpServer = server
        await register_server(
            name=args.zeroconf, port=tcp_server.port, host=tcp_server.host
        )
        _LOGGER.debug("Zeroconf discovery enabled")

    try:
        await server.run(partial(SnowboyEventHandler, args, state))
    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------


class SnowboyEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        cli_args: argparse.Namespace,
        state: State,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.client_id = str(time.monotonic_ns())
        self.state = state
        self.converter = AudioChunkConverter(rate=16000, width=2, channels=1)
        self.client_data = ClientData()

        _LOGGER.debug("Client connected: %s", self.client_id)
        self.state.clients[self.client_id] = self.client_data

    async def handle_event(self, event: Event) -> bool:
        try:
            if Describe.is_type(event.type):
                wyoming_info = self._get_info()
                await self.write_event(wyoming_info.event())
                _LOGGER.debug("Sent info to client: %s", self.client_id)
                return True

            if Detect.is_type(event.type):
                detect = Detect.from_event(event)
                if detect.names:
                    # Load specific requested keywords
                    self._ensure_keywords_loaded(detect.names)
                    self.client_data.active_keywords = set(detect.names)
                else:
                    # If no specific keywords requested, load and activate all available ones
                    self._load_all_keywords()
            elif AudioStart.is_type(event.type):
                # Reset detection state for all detectors
                for detector in self.client_data.detectors.values():
                    detector.is_detected = False

                # Clear audio buffer
                self.client_data.audio_buffer = bytes()

                # If no keywords are loaded yet, load all available ones
                if not self.client_data.detectors:
                    self._load_all_keywords()

                _LOGGER.debug("Receiving audio from client: %s", self.client_id)

            elif AudioChunk.is_type(event.type):
                # If no keywords are loaded yet, load all available ones
                if not self.client_data.detectors:
                    self._load_all_keywords()

                chunk = AudioChunk.from_event(event)
                chunk = self.converter.convert(chunk)
                self.client_data.audio_buffer += chunk.audio

                # Flag to track if we've sent a detection in this chunk processing
                detection_sent = False

                while len(self.client_data.audio_buffer) >= BYTES_PER_CHUNK:
                    chunk_data = self.client_data.audio_buffer[:BYTES_PER_CHUNK]

                    # Process audio through all active detectors
                    for keyword_name in list(self.client_data.active_keywords):
                        if keyword_name not in self.client_data.detectors:
                            continue

                        detector = self.client_data.detectors[keyword_name]
                        if detector.is_detected:
                            continue

                        # Return is:
                        # -2 silence
                        # -1 error
                        #  0 voice
                        #  n index n-1
                        result_index = detector.detector.RunDetection(chunk_data)
                        if result_index > 0:
                            _LOGGER.debug(
                                "Detected %s from client %s",
                                keyword_name,
                                self.client_id,
                            )
                            detector.is_detected = True

                            # Only send one detection per chunk processing to avoid overwhelming the client
                            if not detection_sent:
                                try:
                                    await self.write_event(
                                        Detection(
                                            name=keyword_name, timestamp=chunk.timestamp
                                        ).event()
                                    )
                                    detection_sent = True

                                    # After sending a detection, we can return to let the client process it
                                    # This helps prevent connection reset errors
                                    self.client_data.audio_buffer = (
                                        self.client_data.audio_buffer[BYTES_PER_CHUNK:]
                                    )
                                    return True
                                except (ConnectionResetError, BrokenPipeError) as e:
                                    _LOGGER.warning(
                                        f"Connection error while sending detection: {e}"
                                    )
                                    return False

                    self.client_data.audio_buffer = self.client_data.audio_buffer[
                        BYTES_PER_CHUNK:
                    ]

            elif AudioStop.is_type(event.type):
                # Inform client if no detections occurred
                if not any(
                    detector.is_detected
                    for detector in self.client_data.detectors.values()
                ):
                    # No wake word detections
                    try:
                        await self.write_event(NotDetected().event())
                        _LOGGER.debug(
                            "Audio stopped without detection from client: %s",
                            self.client_id,
                        )
                    except (ConnectionResetError, BrokenPipeError) as e:
                        _LOGGER.warning(
                            f"Connection error while sending NotDetected: {e}"
                        )
                        return False
                return False
            else:
                _LOGGER.debug(
                    "Unexpected event: type=%s, data=%s", event.type, event.data
                )
            return True
        except (ConnectionResetError, BrokenPipeError) as e:
            _LOGGER.warning(f"Connection error in handle_event: {e}")
            return False
        except Exception as e:
            _LOGGER.exception(f"Unexpected error in handle_event: {e}")
            return False

    async def disconnect(self) -> None:
        _LOGGER.debug("Client disconnected: %s", self.client_id)
        # Remove client data
        if self.client_id in self.state.clients:
            del self.state.clients[self.client_id]

    def _ensure_keywords_loaded(self, keyword_names: List[str]):
        """Ensure specific keywords are loaded"""
        for keyword_name in keyword_names:
            if keyword_name not in self.client_data.detectors:
                try:
                    self.client_data.detectors[
                        keyword_name
                    ] = self.state.create_detector(keyword_name)
                    _LOGGER.debug(f"Loaded keyword detector for {keyword_name}")
                except ValueError as e:
                    _LOGGER.warning(f"Failed to load keyword {keyword_name}: {e}")

        if not self.client_data.active_keywords:
            # If no active keywords yet, activate all loaded ones
            self.client_data.active_keywords = set(self.client_data.detectors.keys())

        _LOGGER.debug(
            f"Active keywords for client {self.client_id}: {', '.join(self.client_data.active_keywords)}"
        )

    def _load_all_keywords(self):
        """Load all available keywords"""
        all_keywords = list(self.state.available_keywords.keys())
        self._ensure_keywords_loaded(all_keywords)

    def _get_info(self) -> Info:
        return Info(
            wake=[
                WakeProgram(
                    name="snowboy",
                    description="DNN based hotword and wake word detection toolkit",
                    attribution=Attribution(
                        name="Kitt.AI", url="https://github.com/Kitt-AI/snowboy"
                    ),
                    installed=True,
                    version=__version__,
                    models=[
                        WakeModel(
                            name=kw.name,
                            description=kw.name,
                            phrase=kw.name.replace("_", " "),
                            attribution=Attribution(
                                name="Kitt.AI",
                                url="https://github.com/Kitt-AI/snowboy",
                            ),
                            installed=True,
                            languages=[],
                            version="1.3.0",
                        )
                        for kw in self.state.available_keywords.values()
                    ],
                )
            ],
        )


# -----------------------------------------------------------------------------


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
