"""OpenPixel backend-core: pos-processamento, servicos e API."""
import sys
from pathlib import Path

_CONTRACTS = Path(__file__).resolve().parents[2] / "shared-contracts"
if str(_CONTRACTS) not in sys.path:
    sys.path.insert(0, str(_CONTRACTS))
