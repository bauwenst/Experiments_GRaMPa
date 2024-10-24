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

    # for temperature in [1.0, 1.025, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
    for temperature in [1.0]:
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

    renorm.resetTemperature(1)
    segmentality = LineGraph(f"{tk.getName()}_{corpus.name}_t-vs-segmentality", caching=CacheMode.WRITE_ONLY)
    length       = LineGraph(f"{tk.getName()}_{corpus.name}_t-vs-length", caching=CacheMode.WRITE_ONLY)

    if segmentality.needs_computation or length.needs_computation:
        for t in [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0, -100.0, -50.0, -40.0, -30.0, -20.0, -10.0, -5.0, -4.0, -3.0, -2.0, -1.0, -0.1]:
        # for t in [-1e-15, -1e-14, -1e-13, -1e-12, -1e-11, -1e-10, -1e-9, -1e-8, -1e-7, -1e-6, -1e-5, -1e-4, -1e-3, -1e-2, -1e-1, -1]:
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
            logx_becomes_linear_at=0.1
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


def histogramShiftsWithTemperature(tk: RandomVocabSegmentation_GreedyMarkov, word: str, corpus: NamedIterable[str]):
    n_chars = len("".join(tk.prepareAndTokenise(word)))

    histo_across_amounts = StreamingMultiHistogram(f"amounts_{word}_{tk.getName()}", BinSpec.closedFromAmount(minimum=1, maximum=n_chars+1, amount=n_chars),
                                                   caching=CacheMode.IF_MISSING)
    token_counts = StreamingMultiHistogram(f"count-tokens_{corpus.name}_{tk.getName()}",
                                           BinSpec.halfopen(minimum=0, width=50), caching=CacheMode.IF_MISSING)

    renorm = tk.renormalisation
    assert isinstance(renorm, PowerNormalisation)

    temperatures = [1.0, 5.0, -10.0]

    tk.enableInfiniteDomain(True)
    for t in temperatures:
        name = rf"$\tau = {t}$"
        renorm.resetTemperature(t)
        for _ in streamProgress(range(100_000)):
            tokens = tk.prepareAndTokenise(word)
            histo_across_amounts.add(name, len(tokens))

    histo_across_amounts.commit(StreamingMultiHistogram.ArgsGlobal(
        x_label="Amount of tokens $m$",
        x_tickspacing=1,
        x_center_ticks=True,

        y_label=f"Fraction of samples",
        relative_counts=True
    ))

    tk.enableInfiniteDomain(False)
    for example in streamProgress(corpus):
        for t in temperatures:
            name = rf"$\tau = {t}$"
            renorm.resetTemperature(t)
            for _ in range(100):
                tokens = tk.prepareAndTokenise(example)
                token_counts.add(name, len(tokens))

    token_counts.commit(StreamingMultiHistogram.ArgsGlobal(
        x_tickspacing=256,
        x_label="Amount of tokens",
        x_lims=(None,3_000),

        y_label="Examples in corpus",
        relative_counts=True
    ))


def segmentalityGraph(tk: Tokeniser, word_corpus: NamedIterable[str], n_samples: int=100):
    segmentality = VariableGranularityHistogram(f"{tk.getName()}_{word_corpus.name}_{n_samples}_segmentality", caching=CacheMode.WRITE_ONLY)

    if segmentality.needs_computation:
        for word in word_corpus:
            for _ in range(n_samples):
                tokens = tk.prepareAndTokenise(word)
                segmentality.add(len(tokens)-1, len("".join(tokens)))

    segmentality.commit(VariableGranularityHistogram.ArgsGlobal(
        x_label=r"Segmentality $\mathcal S$ (word tokens $\to$ character tokens)",
        y_label="Fraction of words",
        relative_counts=True,
        x_tickspacing=0.1,
        n_bins=30
    ))



if __name__ == "__main__":
    tk = createTokeniser_SwitchyGrampa_BPE(t=1.0, l=1).subtokenisers[1]
    assert isinstance(tk, RandomVocabSegmentation_GreedyMarkov)
    tk.enableInfiniteDomain(enable=False)

    ###

    from typing import Iterable
    class DummyIterableThatJustMakesGenerators(Iterable):
        def __iter__(self):
            with KnockoutDataConfiguration(setupEnglish()):
                return take(10_000,keepFirst(o.word for o in morphologyGenerator(verbose=True)))
    word_corpus = NamedIterable(DummyIterableThatJustMakesGenerators(), name="celex-en-lemmata")
    segmentalityGraph(tk, word_corpus)

    ###

    _,_,validation = loadCorpus(CORPUS_ID)
    corpus = NamedIterable(validation, name=validation.info.dataset_name).map(lambda example: example["text"])
    word = "antidisestablishmentarianism"
    histogramShiftsWithTemperature(tk, word, corpus)

    # main_GRaMPa_word(word, unconstrained=True)
    # main_BPE_corpus(corpus)
    # main_GRaMPa_corpus(corpus, unconstrained=True)

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