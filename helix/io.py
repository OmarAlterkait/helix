"""Back-compat shim — moved to helix.tpc.io."""
from helix.tpc.io import (
    config_from_file, read_sensor_plane, read_sensor_event,
    count_events, write_processed,
)

__all__ = [
    "config_from_file", "read_sensor_plane", "read_sensor_event",
    "count_events", "write_processed",
]
