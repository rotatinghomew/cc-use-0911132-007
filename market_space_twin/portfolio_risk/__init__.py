"""Tenant portfolio cash-flow risk comparison.

The package deliberately stays inside the Python standard library.  It reads
monthly lease/cash-flow facts, separates contract-basis effects (contract rent
and explicit concessions) from comparable collection performance, estimates
co-movement with a shrunk sector-block correlation model and reports downside
risk for several leasing plans.  Management shock inputs are carried as
labelled assumptions, never as estimated probabilities.
"""

__all__: list[str] = []
