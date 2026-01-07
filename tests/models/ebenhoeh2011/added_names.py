from typing import Literal
from mxlbricks.names import loc

EMPTY: Literal[""] = ""

def total_psii_rc(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    """Total photosystem II reaction centers."""
    return loc("Total PSII RCs", compartment, tissue)

def quencher_active(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    return loc("Q_active", compartment, tissue)

def quencher_inactive(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    return loc("Q_inactive", compartment, tissue)

def total_quencher(compartment: str = EMPTY, tissue: str = EMPTY) -> str:
    """Total quencher."""
    return loc("Total Q", compartment, tissue)