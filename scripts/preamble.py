from pathlib import Path
PATH_ROOT = Path(__file__).resolve().parent.parent
PATH_DATA = PATH_ROOT / "data"
PATH_DATA_OUT = PATH_DATA / "out"
PATH_DATA_OUT.mkdir(parents=True, exist_ok=True)

from fiject import setFijectOutputFolder
setFijectOutputFolder(PATH_DATA_OUT)

from tktkt import setTkTkToutputRoot
setTkTkToutputRoot(PATH_DATA_OUT)

from lamoto.training.training import LamotoPaths
from tktkt.util.environment import is_cluster
