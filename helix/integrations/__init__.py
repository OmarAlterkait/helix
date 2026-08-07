"""Optional adapters that plug helix into a downstream framework.

Nothing here is imported by ``helix`` itself — each module in this package
imports a THIRD-PARTY framework at module scope, so importing one is an explicit
act by a consumer that already has that framework installed. ``import helix``
stays free of every one of them.
"""
