"""helix.tpc — liquid-argon TPC wire pipeline: coherent removal + wavelet."""
from helix.tpc.config import DetectorConfig
from helix.tpc.coherent import remove_coherent
from helix.tpc.pipeline import process_plane, process_event, ProcessedPlane
from helix.tpc.io import config_from_file

__all__ = [
    "DetectorConfig", "remove_coherent",
    "process_plane", "process_event", "ProcessedPlane", "config_from_file",
]
