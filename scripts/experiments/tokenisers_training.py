from scripts.preamble import *

from typing import Tuple
from datasets import load_dataset, IterableDataset, IterableDatasetDict

from tktkt.models.bpe.vocabularisation import BPEVocabulariser, BpeTrainerImplementation
from tktkt.models.kudopiece.vocabularisation import KudoPieceVocabulariser, KudoPieceArguments
from tktkt.factories.preprocessing import ModernEnglishPreprocessor_ByteCompatible, \
    SentencePiecePreprocessor_SpaceConcatenable, RobertaSpaceMarker
from tktkt.util.timing import datetimeDashed
from tktkt.util.environment import IS_NOT_LINUX


TRUNCATE_INPUT_AFTER = 8192

if IS_NOT_LINUX:
    TRAINING_CORPUS_SIZE = 5000
    # VALIDATION_CORPUS_SIZE = 500
    # CORPUS_ID = ("oscar-corpus/oscar", "unshuffled_deduplicated_en")  # Has no validation split
    CORPUS_ID = ("allenai/c4", "en")
    VALIDATION_CORPUS_SIZE = 1_000  # For tuning the GRaMPa hyperparameters.
    # CORPUS_ID = ("cerebras/SlimPajama-627B",)  # Takes 10 minutes to start streaming....
else:
    TRAINING_CORPUS_SIZE = 3_000_000  # Needs >200 GiB RAM for KudoPiece.
    VALIDATION_CORPUS_SIZE = 20_000  # For tuning the GRaMPa hyperparameters.
    CORPUS_ID = ("cerebras/SlimPajama-627B",)  # Takes 10 minutes to start streaming....


VOCAB_SIZE = 32768
MARKER = RobertaSpaceMarker
MAX_LENGTH = 32

def loadCorpus(corpus_id: Tuple[str,...], train_size: int=TRAINING_CORPUS_SIZE, validation_size: int=VALIDATION_CORPUS_SIZE,
               cache=dict()) -> Tuple[IterableDatasetDict, IterableDataset, IterableDataset]:
    """
    Load the corpus lazily and then keep the references to the iterables for if you need to do it again later.
    """
    if corpus_id in cache:
        return cache[corpus_id]

    print(datetimeDashed(), "Loading lazy corpus...")
    corpus_splits: IterableDatasetDict = load_dataset(*corpus_id, streaming=True, trust_remote_code=True)
    print(datetimeDashed(), "Finished loading. Taking sizes", (train_size, validation_size))
    print(corpus_splits)

    train_corpus: IterableDataset = corpus_splits["train"]
    train_corpus: IterableDataset = train_corpus.take(train_size)
    valid_corpus: IterableDataset = corpus_splits["validation"]
    valid_corpus: IterableDataset = valid_corpus.take(validation_size)

    cache[corpus_id] = (corpus_splits, train_corpus, valid_corpus)
    return corpus_splits, train_corpus, valid_corpus


def trainBPE():
    """
    Takes about 5 hours to go through a 750k corpus (50 it/s => 15k seconds) with BPEasy.
    """
    imp = BpeTrainerImplementation.SENTENCEPIECE
    if imp == BpeTrainerImplementation.BPEASY:
        preprocessor = ModernEnglishPreprocessor_ByteCompatible(marker=MARKER)
    else:
        preprocessor = SentencePiecePreprocessor_SpaceConcatenable(MARKER.location)
    vocabulariser = BPEVocabulariser(preprocessor=preprocessor, implementation=imp,
                                     vocab_size=VOCAB_SIZE, max_token_length=MAX_LENGTH,
                                     skip_sentences_over_length=TRUNCATE_INPUT_AFTER)
    _, train_corpus, _ = loadCorpus(CORPUS_ID)
    vocabulariser.vocabulariseFromHf(train_corpus, text_field="text")


def trainKudo():
    """
    Takes about an hour to get through a 750k corpus (250 it/s => 3k seconds).
    Very high memory consumption (80 GiB per million sentences).
    """
    vocabulariser = KudoPieceVocabulariser(
        preprocessor=SentencePiecePreprocessor_SpaceConcatenable(MARKER.location),
        final_vocab_size=VOCAB_SIZE,
        arguments=KudoPieceArguments(
            skip_sentences_over_length=TRUNCATE_INPUT_AFTER,
            maximum_token_length=MAX_LENGTH
        )
    )
    _, train_corpus, _ = loadCorpus(CORPUS_ID)
    vocabulariser.vocabulariseFromHf(train_corpus, text_field="text")


if __name__ == "__main__":
    trainBPE()
    trainKudo()
