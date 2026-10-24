"""Build from extracted SBND files (:mod:`helix.tpc.io_sbnd`, decoded in worker processes): the TPC
readout (adc, pedestal, present per channel), PMT and X-ARAPUCA waveforms. Chain: SBND_DEFAULT.

    python scripts/build_sbnd.py --input RUN/*.h5 --output OUT/RUN/sbnd --device cuda --count 100
    python scripts/build_sbnd.py plan OUT RUN_DIR...       every --events-per-task events of a run

Options and the shards: :mod:`helix.build`.
"""
from __future__ import annotations

from pathlib import Path

from helix.build import main


class SBNDSource:
    """Extracted SBND files: the TPC readout and the PMT / X-ARAPUCA waveforms."""

    systems = ("pmt", "xarapuca")
    dataset = "sbnd"
    pattern = "*.h5"

    def __init__(self, files, device, optical=True):
        from helix.tpc.io_sbnd import SBNDEvents
        self.files, self.device = sorted(files), device
        self.systems = SBNDSource.systems if optical else ()
        self.n = len(SBNDEvents(self.files))

    def __len__(self):
        return self.n

    def chain_config(self, overrides):
        from helix.tpc.chain import make_config
        return make_config(overrides)

    def bind(self, chain):
        pass

    def optical_chain(self, overrides):
        from helix.optical.codec import OpticalChain
        return OpticalChain(overrides, device=self.device)

    def events(self, indices, workers):
        from helix.tpc.io_sbnd import loader
        for ev in loader(self.files, workers=workers, indices=indices, optical_kinds=self.systems):
            m = ev["meta"]
            yield dict(ident=dict(run=str(m["run"]), source_file=Path(m["file"]).name, event=int(m["event"]),
                                  subrun=int(m["subrun"]), time=float(m["time"])),
                       readout=(ev["adc"], ev["pedestal"], ev["present"]),
                       optical={s: ev[s] for s in self.systems})

    def shard_attrs(self):
        return dict(noise=None, provenance=None)


if __name__ == "__main__":
    main(SBNDSource)
