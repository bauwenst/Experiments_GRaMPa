from scripts.preamble import *
# from tst.constants import *  # ---> The constants are not used in this file because EVEN when the constants change, the tokenisers don't.
from scripts.experiments.tokenisers_training import MARKER

from typing import Tuple

from tktkt.interfaces.tokeniser import TokeniserWithFiniteTypeDomain
from tktkt.factories.tokenisers import Factory_BPE, Factory_KudoPiece, Factory_SwitchyGrampa_BPE, Factory_SwitchyGrampa_ULM


def getTokeniserByModelId(model_id: int) -> Tuple[TokeniserWithFiniteTypeDomain, str]:
    """
    Hard-coded tokeniser instances enumerated for easy selection with a command-line script.
    """
    if model_id == 1:  # 150 seconds == 2.5 minutes per batch.
        tokeniser = Factory_BPE(dropout=0.1).buildTokeniser()
        shorthand = "BPE-dropout-0.1"
    elif model_id == 2:
        tokeniser = Factory_KudoPiece(kbest=64, alpha=0.15).buildTokeniser()
        shorthand = "ULM-64-0.15"
    elif model_id in {3, 4, 5, 9}:
        if model_id == 3:
            temperature    = +1.0
            minimum_length = 2
        elif model_id == 4:
            temperature    = +5.0
            minimum_length = 2
        elif model_id == 5:
            temperature    = -10.0
            minimum_length = 2
        elif model_id == 9:
            temperature    = +1.0
            minimum_length = 1
        else:
            raise RuntimeError()

        tokeniser = Factory_SwitchyGrampa_BPE(
            temperature=temperature, l_min=minimum_length,
            p=0.5
        )
        shorthand = f"BPE+GRaMPa(t={temperature},l={minimum_length})"
    elif model_id in {6, 7, 8, 10}:  # 220 seconds == 3.67 minutes per batch.
        if model_id == 6:
            temperature    = +1.0
            minimum_length = 2
        elif model_id == 7:
            temperature    = +5.0
            minimum_length = 2
        elif model_id == 8:
            temperature    = -10.0
            minimum_length = 2
        elif model_id == 10:
            temperature    = +1.0
            minimum_length = 1
        else:
            raise RuntimeError()

        tokeniser = Factory_SwitchyGrampa_ULM(
            kbest=1, smoothing_power=1,
            temperature=temperature, l_min=minimum_length,
            p=0.5
        )
        shorthand = f"ULM+GRaMPa(t={temperature},l={minimum_length})"
    else:
        raise ValueError("Unknown model id:", model_id)

    return tokeniser, shorthand
