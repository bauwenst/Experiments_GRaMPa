"""
TODO: Fully deprecate this file.
"""
from scripts.preamble import *
from tktkt.util.iterables import keepFirst
from tktkt.util.types import NamedIterable

from bpe_knockout import morphologyGenerator, KnockoutDataConfiguration, setupEnglish


if __name__ == "__main__":
    from typing import Iterable
    class DummyIterableThatJustMakesGenerators(Iterable):
        def __iter__(self):
            with KnockoutDataConfiguration(setupEnglish()):
                return keepFirst(o.word for o in morphologyGenerator(verbose=True))
    word_corpus = NamedIterable(DummyIterableThatJustMakesGenerators(), name="celex-en-lemmata")

    # tk = createTokeniser_SwitchyGrampa_BPE(t=1.0, l=1).subtokenisers[1]
    # assert isinstance(tk, RandomVocabSegmentation_GreedyMarkov)
    # tk.enableInfiniteDomain(enable=False)
    # plot_segmentality(tk, word_corpus)

    ###

    # _,_,validation = loadCorpus(CORPUS_ID)
    # corpus = NamedIterable(validation, name=validation.info.dataset_name).map(lambda example: example["text"])
    # word = "antidisestablishmentarianism"
    # plot_histogramShiftsWithTemperature(tk, word, corpus)

    # main_GRaMPa_word(word, unconstrained=True)
    # main_BPE_corpus(corpus)
    # main_GRaMPa_corpus(corpus, unconstrained=True)

    plot_fertilities(tokenisers, raw_words=word_corpus)

    # visualiseCharsVersusTokensRelationships(
    #     tokeniser=tk,
    #     raw_words=corpus,
    #     n_samples_per_word=100
    # )
    # visualiseSingleWordSegmentationDistribution(
    #     tokeniser=tk,
    #     word=word,
    #     samples=100_000,
    #     segmentation_histogram_max_bins=2**9
    # )