from tst.preamble import *
from tst.experiments.tokenisers_training import MARKER

from typing import Tuple

from tktkt.interfaces import Preprocessor, Vocab
from tktkt.interfaces.tokeniser import TokeniserWithFiniteTypeDomain
from tktkt.preparation.instances import ModernEnglishPreprocessor, RobertaSpaceMarker, KudoSpaceMarker, \
    SentencePiecePreprocessor, IdentityPreprocessor, TraditionalPretokeniser, TruncateAndNormalise, IdentityMapper, BoundaryMarker
from tktkt.models.bpe.vocabularisation import BPEVocabulariser, DEFAULT_FIVE_SPECIALS
from tktkt.models.kudopiece.segmentation import KudoPieceTokeniser
from tktkt.models.kudopiece.vocabularisation import KudoPieceTrainer
from tktkt.models.random.pathmarkov import RandomVocabSegmentation_GreedyMarkov, PowerNormalisation
from tktkt.models.huggingface.bpe import HuggingFaceBPETokeniser
from tktkt.wrappers.multiplexing import StochasticTokeniserSwitch, MultiplexedPreprocessor
from tktkt.files.paths import TkTkTPaths
from tktkt.builders.base import TokeniserBuilder


TRUNCATE_INPUT_AFTER = 8192


class Build_GRaMPa(TokeniserBuilder[RandomVocabSegmentation_GreedyMarkov]):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, minimal_length: int=2, temperature: float=1.0):
        self._prep = preprocessor
        self._vocab = vocab
        self._temp = temperature
        self._minlen = minimal_length

    def buildTokeniser(self):
        return RandomVocabSegmentation_GreedyMarkov(
            preprocessor=self._prep,
            vocab=self._vocab,
            unk_type=DEFAULT_FIVE_SPECIALS.unk_token,

            probabilities_to_probabilities=PowerNormalisation(temperature=self._temp),
            minimal_token_length=self._minlen,
            decode_backwards=False
        )


class Build_English_BPE(TokeniserBuilder[HuggingFaceBPETokeniser]):
    """
    Defaults to the 32k SlimPajama vocab.
    """

    def __init__(self, preprocessor: Preprocessor=None, path: Path=None, dropout: float=0.0):
        if path is None:
            path = TkTkTPaths.pathToModels() / "bpe" / "bpe_slim_pajama-627_b_2024-10-06_02-40-55"
        if preprocessor is None:
            preprocessor = ModernEnglishPreprocessor(marker=MARKER, truncate_text_after_chars=TRUNCATE_INPUT_AFTER)

        self._prep = preprocessor
        self._path = path
        self._dropout = dropout

        self.vocab  = BPEVocabulariser.load(self._path, existing_types=DEFAULT_FIVE_SPECIALS.all_special_tokens)
        self.merges = BPEVocabulariser.loadMerges(self._path)

    def buildTokeniser(self):
        return HuggingFaceBPETokeniser(
            vocab=self.vocab,
            merges=self.merges,
            dropout=self._dropout,
            preprocessor=self._prep
        )


class Build_English_Kudo(TokeniserBuilder[KudoPieceTokeniser]):
    """
    Defaults to the 32k SlimPajama vocab.
    """

    def __init__(self, preprocessor: Preprocessor=None, folder: Path=None, kbest: int=64, alpha: float=1.0):
        if folder is None:
            folder = TkTkTPaths.pathToModels() / "kudopiece" / "kudopiece_slim_pajama-627_b_2024-10-06_11-26-39"
        if preprocessor is None:
            preprocessor = SentencePiecePreprocessor(marker=MARKER, prefix_space_already_added=True)  # Marker is only used for its location. I fucked up and set add_prefix to True when training the tokeniser, and now that option is baked into the .model file LMAO.

        self._prep = preprocessor
        self._folder = folder
        self._kbest = kbest
        self._alpha = alpha

    def buildTokeniser(self):
        vocab = KudoPieceTrainer.load(self._folder, existing_types=DEFAULT_FIVE_SPECIALS.all_special_tokens)
        return KudoPieceTokeniser(
            preprocessor=self._prep,
            model_file=self._folder / "spm.model",
            vocab=vocab,

            kbest=self._kbest,
            smoothing_power=self._alpha
        )


def createTokeniser_SwitchyGrampa_ULM(p: float=0.5,
                                      t: float=1.0, l: int=1,
                                      kbest: int=1, smoothing_power: float=1.0) -> StochasticTokeniserSwitch:
    global_preprocessor = Preprocessor(TruncateAndNormalise(TRUNCATE_INPUT_AFTER), IdentityMapper(), TraditionalPretokeniser())

    tk1 = Build_English_Kudo(kbest=kbest, alpha=smoothing_power).buildTokeniser()
    tk2 = Build_GRaMPa(
        preprocessor=ModernEnglishPreprocessor(marker=KudoSpaceMarker, truncate_text_after_chars=TRUNCATE_INPUT_AFTER),
        vocab=tk1.vocab,
        minimal_length=l,
        temperature=t
    ).buildTokeniser()

    return StochasticTokeniserSwitch(
        preprocessor=MultiplexedPreprocessor(
            global_preprocessor=global_preprocessor,
            specific_preprocessors=True
        ),
        tokeniser1=tk1,
        tokeniser2=tk2,
        p=p
    )


def createTokeniser_SwitchyGrampa_BPE(p: float=0.5,
                                      t: float=1.0, l: int=1,
                                      dropout: float=0.0) -> StochasticTokeniserSwitch:
    global_preprocessor = Preprocessor(TruncateAndNormalise(TRUNCATE_INPUT_AFTER), IdentityMapper(), TraditionalPretokeniser())
    sub_preprocessor = ModernEnglishPreprocessor(marker=MARKER, truncate_text_after_chars=TRUNCATE_INPUT_AFTER)

    build1 = Build_English_BPE(preprocessor=sub_preprocessor, dropout=dropout)
    tk1 = build1.buildTokeniser()
    tk2 = Build_GRaMPa(preprocessor=sub_preprocessor, vocab=build1.vocab, minimal_length=l, temperature=t).buildTokeniser()
    return StochasticTokeniserSwitch(
        preprocessor=MultiplexedPreprocessor(
            global_preprocessor=global_preprocessor,
            specific_preprocessors=True
        ),
        tokeniser1=tk1,
        tokeniser2=tk2,
        p=p
    )


def getTokeniserByModelId(model_id: int) -> Tuple[TokeniserWithFiniteTypeDomain, str]:
    """
    Hard-coded tokeniser instances enumerated for easy selection with a command-line script.
    """
    if model_id == 1:  # 150 seconds == 2.5 minutes per batch.
        tokeniser = Build_English_BPE(dropout=0.1).buildTokeniser()
        shorthand = "BPE-dropout"
    elif model_id == 2:
        tokeniser = Build_English_Kudo(kbest=64, alpha=0.15).buildTokeniser()
        shorthand = "ULM"
    elif model_id in {3, 4, 5}:
        if model_id == 3:
            temperature    = 1.0
            minimum_length = 1
        elif model_id == 4:
            temperature    = +5.0
            minimum_length = 2
        elif model_id == 5:
            temperature    = -10.0
            minimum_length = 2
        else:
            raise RuntimeError()

        tokeniser = createTokeniser_SwitchyGrampa_BPE(
            t=temperature, l=minimum_length,
            p=0.5
        )
        shorthand = f"BPE+GRaMPa(t={temperature},l={minimum_length})"
    elif model_id in {6, 7, 8}:  # 220 seconds == 3.67 minutes per batch.
        if model_id == 6:
            temperature    = 1.0
            minimum_length = 1
        elif model_id == 7:
            temperature    = +5.0
            minimum_length = 2
        elif model_id == 8:
            temperature    = -10.0
            minimum_length = 2
        else:
            raise RuntimeError()

        tokeniser = createTokeniser_SwitchyGrampa_ULM(
            kbest=1, smoothing_power=1,
            t=temperature, l=minimum_length,
            p=0.5
        )
        shorthand = f"ULM+GRaMPa(t={temperature},l={minimum_length})"
    else:
        raise ValueError("Unknown model id:", model_id)

    return tokeniser, shorthand
