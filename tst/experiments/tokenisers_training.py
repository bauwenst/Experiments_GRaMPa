from tst.preamble import *

from datasets import load_dataset, IterableDataset, IterableDatasetDict
from tktkt.util.timing import datetimeDashed
from tktkt.util.environment import IS_LINUX
from tktkt.preparation.instances import *
from tktkt.models.bpe.vocabularisation import BPEVocabulariser, BpeTrainerImplementation
from tktkt.models.kudopiece.vocabularisation import KudoPieceTrainer, KudoPieceArguments_Algorithm, KudoPieceArguments_Alphabet


if IS_LINUX:
    CORPUS_SIZE = 500_000  # For BPEasy, my RAM runs out at 898 304 which is the 5hr22min mark. At 750k, it can load everything, but eats 100% of RAM after that. For Unigram, it is able to load 750k, but then after a while crashes silently with an (0xC0000409) error.
    CORPUS_ID = ("cerebras/SlimPajama-627B",)  # Takes 10 minutes to start streaming....
else:
    CORPUS_SIZE = 1000
    CORPUS_ID = ("oscar-corpus/oscar", "unshuffled_deduplicated_en")


VOCAB_SIZE = 32768
MARKER = RobertaSpaceMarker
MAX_LENGTH = 32

def loadCorpus(corpus_id: Tuple[str,...], cache=dict()) -> Tuple[IterableDatasetDict, IterableDataset]:
    """
    Load the corpus lazily and then keep the references to the iterables for if you need to do it again later.
    """
    if corpus_id in cache:
        return cache[corpus_id]

    print(datetimeDashed(), "Loading lazy corpus...")
    corpus_splits: IterableDatasetDict = load_dataset(*corpus_id, streaming=True, trust_remote_code=True)
    train_corpus: IterableDataset = corpus_splits["train"]
    train_corpus: IterableDataset = train_corpus.take(CORPUS_SIZE)
    print(datetimeDashed(), "Finished loading.", CORPUS_SIZE, "sentences will be used for training.")

    cache[corpus_id] = (corpus_splits, train_corpus)
    return corpus_splits, train_corpus


def trainBPE():
    """
    Takes about 5 hours to go through a 750k corpus (50 it/s => 15k seconds) with BPEasy.
    """
    imp = BpeTrainerImplementation.SENTENCEPIECE
    if imp == BpeTrainerImplementation.BPEASY:
        preprocessor = ModernEnglishPreprocessor_ByteCompatible(marker=MARKER)
    else:
        preprocessor = SentencePiecePreprocessor(marker=MARKER)
    vocabulariser = BPEVocabulariser(preprocessor=preprocessor, implementation=imp,
                                     boundary_marker=MARKER, byte_based=True,
                                     vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH)
    _, train_corpus = loadCorpus(CORPUS_ID)
    vocabulariser.vocabulariseFromHf(train_corpus, text_field="text")


def trainKudo():
    """
    Takes about an hour to get through a 750k corpus (250 it/s => 3k seconds).
    """
    vocabulariser = KudoPieceTrainer(
        preprocessor=SentencePiecePreprocessor(marker=MARKER),
        final_vocab_size=VOCAB_SIZE, word_boundary_location=MARKER.location,
        alphabet_arguments=KudoPieceArguments_Alphabet(
            required_chars=[k for k in PseudoByteMapping.PSEUDO_TO_BYTE if k != " "],  # Exclude that space if you want to stay alive https://github.com/google/sentencepiece/issues/1059
            byte_fallback=False, character_coverage=1.0  # We will be using HF coding, remember.
        ),
        algorithm_arguments=KudoPieceArguments_Algorithm(
            maximum_token_length=MAX_LENGTH,
            skip_sentences_over_length=2**13
        )
    )
    _, train_corpus = loadCorpus(CORPUS_ID)
    vocabulariser.vocabulariseFromHf(train_corpus, text_field="text")


if __name__ == "__main__":
    try:
        trainKudo()
    except:
        print("Kudo crashed.")
        pass
    try:
        trainBPE()
    except:
        print("BPE crashed.")
        pass

    import bpe_knockout