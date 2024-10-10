from tst.preamble import *
from tst.experiments.tokenisers_instances import *
from tst.experiments.tokenisers_training import loadCorpus, CORPUS_ID

from wiat.visualisation.tokenisers import *

from tktkt.models.random.pathmarkov import RandomVocabSegmentation_GreedyMarkov, PowerNormalisation
from tktkt.util.iterables import keepFirst, take
from tktkt.util.types import NamedIterable

from bpe_knockout import morphologyGenerator, KnockoutDataConfiguration, setupEnglish

from fiject import LineGraph


def main_BPE_corpus(word_corpus: NamedIterable[str]):
    tk = Build_English_BPE().buildTokeniser()
    visualiseCharsVersusTokensRelationships(
        tokeniser=tk,
        raw_words=word_corpus,
        n_samples_per_word=1
    )


def main_GRaMPa_word(word: str, unconstrained: bool=True):
    tk = createTokeniser_SwitchyGrampa_BPE(l=2).subtokenisers[1]
    assert isinstance(tk, RandomVocabSegmentation_GreedyMarkov)
    renorm = tk.renormalisation
    assert isinstance(renorm, PowerNormalisation)

    # Infinite domain so that we can measure the effect of temperature in a histogram with LOC ordering without weird shit.
    tk.enableInfiniteDomain(unconstrained)

    for temperature in [1.0, 1.025, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
        renorm.resetTemperature(temperature)
        visualiseSingleWordSegmentationDistribution(
            tokeniser=tk,
            word=word,
            samples=100_000,  # TODO: Recommended for official graphs is 500_000
            segmentation_histogram_max_bins=2**9,
            do_bitbased_ordering=False
        )


def main_GRaMPa_corpus(corpus: NamedIterable[str], unconstrained: bool=True):
    tk = createTokeniser_SwitchyGrampa_BPE(l=2).subtokenisers[1]
    assert isinstance(tk, RandomVocabSegmentation_GreedyMarkov)
    renorm = tk.renormalisation
    assert isinstance(renorm, PowerNormalisation)

    # Infinite domain so that we can measure the effect of temperature in a histogram with LOC ordering without weird shit.
    tk.enableInfiniteDomain(unconstrained)

    renorm.resetTemperature(0)
    segmentality = LineGraph(f"{tk.getName()}_{corpus.name}_t-vs-segmentality", caching=CacheMode.WRITE_ONLY)
    length       = LineGraph(f"{tk.getName()}_{corpus.name}_t-vs-length", caching=CacheMode.WRITE_ONLY)

    if segmentality.needs_computation or length.needs_computation:
        # for t in [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0, -100.0, -50.0, -40.0, -30.0, -20.0, -10.0, -5.0, -4.0, -3.0, -2.0, -1.0]:
        for t in [-1e-15, -1e-14, -1e-13, -1e-12, -1e-11, -1e-10, -1e-9, -1e-8, -1e-7, -1e-6, -1e-5, -1e-4, -1e-3, -1e-2, -1e-1, -1]:
            renorm.resetTemperature(t)
            (seg_mean, seg_mode, seg_std), (length_mean, length_mode, length_std) = visualiseCharsVersusTokensRelationships(
                tokeniser=tk,
                raw_words=corpus,
                n_samples_per_word=100
            )

            segmentality.add("mean", t, seg_mean)
            # segmentality.add("mode", t, seg_mode)
            segmentality.add("std", t, seg_std)

            length.add("mean", t, length_mean)
            # length.add("mode", t, length_mode)
            length.add("std", t, length_std)

    segmentality.commitWithArgs(
        LineGraph.ArgsGlobal(
            x_label="Temperature",
            y_label="Segmentality",
            logx=True,
            logx_becomes_linear_at=1e-15 #1.0  # TODO: TEMPORARY
        ),
        LineGraph.ArgsPerLine()
    )
    length.commitWithArgs(
        LineGraph.ArgsGlobal(
            x_label="Temperature",
            y_label="Token length",
            logx=True,
            logx_becomes_linear_at=1e-15
        ),
        LineGraph.ArgsPerLine()
    )



if __name__ == "__main__":
    from typing import Iterable
    class DummyIterableThatJustMakesGenerators(Iterable):
        def __iter__(self):
            with KnockoutDataConfiguration(setupEnglish()):
                return take(500,keepFirst(o.word for o in morphologyGenerator(verbose=True)))
    corpus = NamedIterable(DummyIterableThatJustMakesGenerators(), name="celex-en-lemmata")

    # _,_,validation = loadCorpus(CORPUS_ID)
    # corpus = NamedIterable(validation, name=validation.info.dataset_name).map(lambda example: example["text"])

    # main_GRaMPa_word("antidisestablishmentarianism", unconstrained=False)
    # main_BPE_corpus(corpus)
    main_GRaMPa_corpus(corpus, unconstrained=True)
