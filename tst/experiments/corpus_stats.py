from tst.preamble import *

import datasets

from tktkt.util.iterables import take
from tktkt.builders.english import Builder_English_BPE_native
from fiject import StreamingMultiHistogram, BinSpec, CacheMode

from tqdm.auto import tqdm

from wiat.eda.sentences import tokenCounts
from tst.experiments.tokenisers_training import loadCorpus
from tst.experiments.tokenisers_instances import createTokeniser_SwitchyGrampa_BPE

# CORPUS_ID   = ("allenai/c4", "en")
# CORPUS_SIZE = 2_000_000
CORPUS_ID = ("cerebras/SlimPajama-627B",)
CORPUS_SIZE = 20_000


def histogramOfTokenCounts():
    """
    Generate a histogram of the amount of BPE tokens to expect per example in the dataset.
    """
    # Load tokeniser
    tk = createTokeniser_SwitchyGrampa_BPE().subtokenisers[0]

    # Graphing
    db_name = '/'.join(CORPUS_ID).replace('/', '-')
    tk_name = tk.getName()
    h = StreamingMultiHistogram(f"tokencounts_{db_name}-{CORPUS_SIZE}_{tk_name}",
                                BinSpec.halfopen(minimum=0, width=50), caching=CacheMode.IF_MISSING)
    if h.needs_computation:
        _, train, _ = loadCorpus(CORPUS_ID, train_size=CORPUS_SIZE)
        corpus = tqdm(take(CORPUS_SIZE, (example["text"] for example in train)), total=CORPUS_SIZE)

        for count in tokenCounts(tk, corpus):
            h.add(tk_name, count)

    h.commit(StreamingMultiHistogram.ArgsGlobal(
        x_tickspacing=256,
        x_label="Tokens per example",
        y_label="Examples",
        x_lims=(None,3_000)
    ))


if __name__ == "__main__":
    histogramOfTokenCounts()
