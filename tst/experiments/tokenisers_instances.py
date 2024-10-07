from tktkt.preparation.mappers import IdentityMapper

from tst.preamble import *
from tst.experiments.tokenisers_training import MARKER

from tktkt.interfaces import Preprocessor, Vocab
from tktkt.preparation.instances import ModernEnglishPreprocessor, RobertaSpaceMarker, KudoSpaceMarker, \
    SentencePiecePreprocessor, IdentityPreprocessor, TraditionalPretokeniser, TruncateAndNormalise
from tktkt.models.bpe.vocabularisation import BPEVocabulariser, DEFAULT_FIVE_SPECIALS
from tktkt.models.kudopiece.segmentation import KudoPieceTokeniser
from tktkt.models.kudopiece.vocabularisation import KudoPieceTrainer
from tktkt.models.random.pathmarkov import RandomVocabSegmentation_GreedyMarkov, PowerNormalisation
from tktkt.models.huggingface.bpe import HuggingFaceBPETokeniser
from tktkt.wrappers.multiplexing import StochasticTokeniserSwitch, MultiplexedPreprocessor
from tktkt.files.paths import TkTkTPaths


TRUNCATE_INPUT_AFTER = 8192


def createTokeniser_Grampa(preprocessor: Preprocessor, vocab: Vocab) -> RandomVocabSegmentation_GreedyMarkov:
    return RandomVocabSegmentation_GreedyMarkov(
        preprocessor=preprocessor,
        vocab=vocab,
        unk_type=DEFAULT_FIVE_SPECIALS.unk_token,

        probabilities_to_probabilities=PowerNormalisation(temperature=1.0),  # TODO: Comes from hyperparameter search.
        minimal_token_length=2
    )


def createTokeniser_SwitchyGrampa_ULM() -> StochasticTokeniserSwitch:
    tk_path = TkTkTPaths.pathToModels() / "kudopiece" / "kudopiece_slim_pajama-627_b_2024-10-06_11-26-39"
    vocab = KudoPieceTrainer.load(tk_path, existing_types=DEFAULT_FIVE_SPECIALS.all_special_tokens)

    global_preprocessor = Preprocessor(TruncateAndNormalise(TRUNCATE_INPUT_AFTER), IdentityMapper(), TraditionalPretokeniser())
    preprocessor1 = SentencePiecePreprocessor(marker=MARKER, prefix_space_already_added=True)  # Marker is only used for its location. I fucked up and set add_prefix to True when training the tokeniser, and now that option is baked into the .model file LMAO.
    preprocessor2 = ModernEnglishPreprocessor(marker=KudoSpaceMarker, truncate_text_after_chars=TRUNCATE_INPUT_AFTER)
    return StochasticTokeniserSwitch(
        preprocessor=MultiplexedPreprocessor(
            global_preprocessor=global_preprocessor,
            specific_preprocessors=True
        ),
        tokeniser1=KudoPieceTokeniser(preprocessor1, tk_path / "spm.model", vocab=vocab, kbest=64, smoothing_power=0.5),
        tokeniser2=createTokeniser_Grampa(preprocessor2, vocab),
        p=0.5  # TODO: Comes from hyperparameter search.
    )


def createTokeniser_SwitchyGrampa_BPE() -> StochasticTokeniserSwitch:
    tk_path = TkTkTPaths.pathToModels() / "bpe" / "bpe_slim_pajama-627_b_2024-10-06_02-40-55"
    vocab  = BPEVocabulariser.load(tk_path, existing_types=DEFAULT_FIVE_SPECIALS.all_special_tokens)
    merges = BPEVocabulariser.loadMerges(tk_path)

    global_preprocessor = Preprocessor(TruncateAndNormalise(TRUNCATE_INPUT_AFTER), IdentityMapper(), TraditionalPretokeniser())
    sub_preprocessor = ModernEnglishPreprocessor(marker=MARKER, truncate_text_after_chars=TRUNCATE_INPUT_AFTER)

    hf = HuggingFaceBPETokeniser(vocab, merges, dropout=0.0, preprocessor=sub_preprocessor)
    return StochasticTokeniserSwitch(
        preprocessor=MultiplexedPreprocessor(
            global_preprocessor=global_preprocessor,
            specific_preprocessors=True
        ),
        tokeniser1=hf,
        tokeniser2=createTokeniser_Grampa(sub_preprocessor, vocab),
        p=0.5  # TODO: Comes from hyperparameter search.
    )


if __name__ == "__main__":
    switch = createTokeniser_SwitchyGrampa_ULM()
    sentence = "workhorses, unité!"
    for _ in range(10):
        print("Global preprocessor:", switch.preprocessor.do(sentence))
        print("Tokenised result:", switch.prepareAndTokenise(sentence))
        print()
