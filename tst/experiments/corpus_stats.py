from tst.preamble import *

import datasets

from tktkt.util.iterables import take
from tktkt.builders.english import Builder_English_BPE_native
from fiject import StreamingMultiHistogram, BinSpec, CacheMode

from tqdm.auto import tqdm

from wiat.eda.sentences import tokenCounts


def test_c4():
    """
    Generate a histogram of the amount of BPE tokens to expect per example in C4.
    """

    # Load data
    N = 2_000_000
    DATASET_NAME = ("allenai/c4", "en")

    # Load tokeniser
    tk = Builder_English_BPE_native().buildTokeniser()
    tk_name = tk.getName()

    # Graphing
    h = StreamingMultiHistogram(f"tokencounts_{'/'.join(DATASET_NAME).replace('/', '-')}-{N}_{tk_name}",
                                BinSpec.halfopen(minimum=0, width=50), caching=CacheMode.IF_MISSING)
    if h.needs_computation:
        dataset = datasets.load_dataset(*DATASET_NAME, streaming=True)
        corpus = tqdm(take(N, (example["text"] for example in dataset["train"])), total=N)

        for count in tokenCounts(tk, corpus):
            h.add(tk_name, count)

    h.commit(StreamingMultiHistogram.ArgsGlobal(
        x_tickspacing=1000,
        x_label="Tokens per example",
        y_label="Examples",
        x_lims=(None,3_000)
    ))


if __name__ == "__main__":
    test_c4()
