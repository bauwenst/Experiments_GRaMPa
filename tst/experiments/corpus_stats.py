from tst.preamble import *
from wiat.visualisation.sentences import *

from tktkt.util.iterables import take, streamProgress
from tktkt.preparation.instances import TraditionalPretokeniser, IdentityMapper, Preprocessor

from tst.experiments.tokenisers_training import loadCorpus
from tst.experiments.tokenisers_instances import createTokeniser_SwitchyGrampa_BPE

CORPUS_ID   = ("allenai/c4", "en")
# CORPUS_SIZE = 2_000_000
# CORPUS_ID = ("cerebras/SlimPajama-627B",)
CORPUS_SIZE = 2_000


def main_BPE():
    # Load tokeniser and word isolator
    tk = createTokeniser_SwitchyGrampa_BPE(dropout=0.0).subtokenisers[0]
    prep = Preprocessor(IdentityMapper(), IdentityMapper(), TraditionalPretokeniser())

    # Load corpus
    _, train, _ = loadCorpus(CORPUS_ID, train_size=CORPUS_SIZE)
    corpus = NamedIterable(
        iterable=streamProgress(take(CORPUS_SIZE, (example["text"] for example in train)), known_size=CORPUS_SIZE),
        name='/'.join(CORPUS_ID).replace('/', '-') + f"-{CORPUS_SIZE}"
    )

    visualiseCounts(corpus, tk, prep)


if __name__ == "__main__":
    main_BPE()
