"""
What is the distribution of the amount of tokens produced across sentences of a corpus?
And how does it compare to the maximum input length of a model?
"""
from typing import List

from tktkt.interfaces import Tokeniser, Preprocessor
from tktkt.util.types import NamedIterable

from fiject import StreamingMultiHistogram, BinSpec, CacheMode, FIJECT_DEFAULTS


def tokenLengths(tokeniser: Tokeniser, text: str) -> List[int]:
    return list(map(len, tokeniser.prepareAndTokenise(text)))


def pretokenLengths(preprocessor: Preprocessor, text: str) -> List[int]:
    return list(map(len, preprocessor.do(text)))


def visualiseCounts(corpus: NamedIterable[str], tokeniser: Tokeniser, preprocessor: Preprocessor=None):
    """
    Generate the following histograms:
        - Length of each item in characters;
        - Amount of pretokens in each item;
        - Amount of tokens in each item;
        - Length of each token;
        - Length of each pretoken (usually the same as a word);
    """
    if preprocessor is None:
        preprocessor = tokeniser.preprocessor

    FIJECT_DEFAULTS.GLOBAL_STEM_SUFFIX = f"{corpus.name}_{tokeniser.getName()}"

    # text_length = StreamingMultiHistogram("count-chars",
    #                                       BinSpec.halfopen(minimum=0, width=50), caching=CacheMode.IF_MISSING)
    # word_counts  = StreamingMultiHistogram("count-words",
    #                                        BinSpec.halfopen(minimum=0, width=50), caching=CacheMode.IF_MISSING)
    token_counts = StreamingMultiHistogram("count-tokens",
                                           BinSpec.halfopen(minimum=0, width=50), caching=CacheMode.IF_MISSING)
    # word_lengths = StreamingMultiHistogram("length-words",
    #                                        BinSpec.halfopen(minimum=0, width=1), caching=CacheMode.IF_MISSING)
    token_lengths = StreamingMultiHistogram("length-tokens",
                                           BinSpec.halfopen(minimum=0, width=1), caching=CacheMode.IF_MISSING)
    # graphs = [text_length, word_counts, token_counts, word_lengths, token_lengths]
    graphs = [token_counts, token_lengths]

    if any(g.needs_computation for g in graphs):
        for text in corpus:
            L_t = tokenLengths(tokeniser, text)
            L_w = pretokenLengths(preprocessor, text)

            # text_length.add("", len(text))
            # word_counts.add("", len(L_w))
            token_counts.add("", len(L_t))
            # for l in L_w:
            #     word_lengths.add("", l)
            for l in L_t:
                token_lengths.add("", l)


    # text_length.commit(StreamingMultiHistogram.ArgsGlobal(
    #     x_tickspacing=2500,
    #     x_label="Amount of characters",
    #     x_lims=(None,25_000),
    #
    #     y_label="Examples",
    #     relative_counts = True
    # ))
    # word_counts.commit(StreamingMultiHistogram.ArgsGlobal(
    #     x_tickspacing=256,
    #     x_label="Amount of pretokens",
    #     x_lims=(None,3_000),
    #
    #     y_label="Examples",
    #     relative_counts = True
    # ))
    token_counts.commit(StreamingMultiHistogram.ArgsGlobal(
        x_tickspacing=256,
        x_label="Amount of tokens",
        x_lims=(None,3_000),

        y_label="Examples",
        relative_counts = True
    ))
    # word_lengths.commit(StreamingMultiHistogram.ArgsGlobal(
    #     x_label="Pretoken length",
    #     x_tickspacing=1,
    #     x_center_ticks=True,
    #     x_lims=(1,32),
    #
    #     y_label="Pretokens",
    #     relative_counts=True
    # ))
    token_lengths.commit(StreamingMultiHistogram.ArgsGlobal(
        x_label="Token length",
        x_tickspacing=1,
        x_center_ticks=True,
        x_lims=(1,32),

        y_label="Tokens",
        relative_counts=True
    ))

    FIJECT_DEFAULTS.GLOBAL_STEM_SUFFIX = ""
