"""Back-compat shim — TPC CLI moved to helix.tpc.run."""
from helix.tpc.run import main

if __name__ == "__main__":
    main()
