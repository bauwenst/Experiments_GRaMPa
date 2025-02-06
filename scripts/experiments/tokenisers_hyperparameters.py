"""
Experiments to do with varying the hyperparameters of the tokenisers.
"""
from scripts.preamble import *
from scripts.experiments.lineages import *
from scripts.experiments.tokenisers_training import loadCorpus, CORPUS_ID

from typing import Iterable, Tuple
import numpy as np

from tktkt.evaluation.entropy import renyiEfficiency, tokenDistributionFromSentences
from tktkt.evaluation.compare import ExactMatches
from tktkt.visualisation.charts.token_distributions import visualiseCharsVersusTokensRelationships, visualiseSingleWordSegmentationDistribution
from tktkt.models.random.pathmarkov import PowerNormalisation, GRaMPa
from tktkt.models.kudopiece.segmentation import KudoPieceTokeniser
from tktkt.wrappers.multiplexing import StochasticTokeniserSwitch
from tktkt.factories.preprocessing import TraditionalPreprocessor
from tktkt.factories.tokenisers import Factory_GRaMPa
from tktkt.util.types import NamedIterable
from tktkt.util.printing import wprint
from tktkt.util.environment import IS_NOT_LINUX
from tktkt.util.iterables import streamProgress

from fiject import LineGraph, CacheMode, StreamingMultiHistogram, BinSpec

from matplotlib import rc
rc("text.latex", preamble=r"\DeclareUnicodeCharacter{2581}{\_}")

LOW_KEY  = r"$H_\alpha/\lceil H_0\rceil$"
MID_KEY  = r"$H_\alpha/H_0$"
HIGH_KEY = r"$\lceil H_\alpha \rceil/H_0$"


def searchTemperatures(markov_tokeniser: GRaMPa, corpus: NamedIterable[str], temperature_grid: Iterable[float]) -> Tuple[float,float]:
    normaliser = markov_tokeniser.renormalisation
    assert isinstance(normaliser, PowerNormalisation)
    normaliser.resetTemperature(0)
    g = LineGraph(f"renyi_t_{markov_tokeniser.getName()}_{corpus.name}", caching=CacheMode.WRITE_ONLY)
    if g.needs_computation:
        for t in temperature_grid:
            wprint(f"Now testing temperature t={t}...")
            normaliser.resetTemperature(t)

            unigram_distribution = tokenDistributionFromSentences(markov_tokeniser, corpus)
            low, mid, high = renyiEfficiency(probabilities=unigram_distribution.values(), alpha=2.5)
            wprint(low, mid, high)

            g.add(LOW_KEY,  t, low)
            g.add(MID_KEY,  t, mid)
            g.add(HIGH_KEY, t, high)

    g.commitWithArgs(LineGraph.ArgsGlobal(
        y_lims=(0.30,0.60),
        x_label=r"Temperature $\tau$",
        y_label="Rényi efficiency bounds",
        legend_position="lower right"
    ), LineGraph.ArgsPerLine())

    ts, lows = g.data[LOW_KEY]
    idx_argmax = np.argmax(lows)
    return ts[idx_argmax], lows[idx_argmax]


def searchMultiplexP(multiplex_tokeniser: StochasticTokeniserSwitch, corpus: NamedIterable[str], probability_grid: Iterable[float]):
    g = LineGraph(f"renyi_p_{multiplex_tokeniser.subtokenisers[0].getName()}+{multiplex_tokeniser.subtokenisers[1].getName()}_{corpus.name}", caching=CacheMode.WRITE_ONLY)
    if g.needs_computation:
        for p in probability_grid:
            wprint(f"Now testing with {multiplex_tokeniser.subtokenisers[1].getName()} at p={p}...")
            multiplex_tokeniser.threshold = p

            unigram_distribution = tokenDistributionFromSentences(multiplex_tokeniser, corpus)
            low, mid, high = renyiEfficiency(probabilities=unigram_distribution.values(), alpha=2.5)
            wprint(low, mid, high)

            g.add(LOW_KEY,  p, low)
            g.add(MID_KEY,  p, mid)
            g.add(HIGH_KEY, p, high)

    g.commitWithArgs(LineGraph.ArgsGlobal(
        y_lims=(0.30,0.60),
        x_label="Multiplexing $p$",
        y_label="Rényi efficiency bounds",
        x_tickspacing=0.1,
        legend_position="upper left"
    ), LineGraph.ArgsPerLine())

    ps, lows = g.data[LOW_KEY]
    idx_argmax = np.argmax(lows)
    return ps[idx_argmax], lows[idx_argmax]


def searchKudoAlpha(kudo_tokeniser: KudoPieceTokeniser, corpus: NamedIterable[str], alpha_grid: Iterable[float]) -> Tuple[float,float]:
    kudo_tokeniser._alpha = "a"  # Trick to get the name independent.
    g = LineGraph(f"renyi_α_{kudo_tokeniser.getName()}_{corpus.name}", caching=CacheMode.WRITE_ONLY)
    if g.needs_computation:
        for a in alpha_grid:
            wprint(f"Now testing alpha a={a}...")
            kudo_tokeniser._alpha = a

            unigram_distribution = tokenDistributionFromSentences(kudo_tokeniser, corpus)
            low, mid, high = renyiEfficiency(probabilities=unigram_distribution.values(), alpha=2.5)
            wprint(low, mid, high)

            g.add(LOW_KEY,  a, low)
            g.add(MID_KEY,  a, mid)
            g.add(HIGH_KEY, a, high)

    g.commitWithArgs(LineGraph.ArgsGlobal(
        # y_lims=(0.35,0.65),
        x_label=r"ULM normalisation power $\alpha$",
        x_tickspacing=0.1,
        y_label="Rényi efficiency bounds",
        legend_position="lower right"
    ), LineGraph.ArgsPerLine())

    alphas, lows = g.data[LOW_KEY]
    idx_argmax = np.argmax(lows)
    return alphas[idx_argmax], lows[idx_argmax]


def searchDropout(corpus: NamedIterable[str], dropout_grid: Iterable[float]) -> Tuple[float,float]:
    tk = Factory_BPE(dropout=0).buildTokeniser()
    tk._dropout_as_string = "p"
    g = LineGraph(f"renyi_dropout_{tk.getName()}_{corpus.name}", caching=CacheMode.WRITE_ONLY)
    if g.needs_computation:
        for p in dropout_grid:
            wprint(f"Now testing dropout p={p}...")
            tk = Factory_BPE(dropout=p).buildTokeniser()

            unigram_distribution = tokenDistributionFromSentences(tk, corpus)
            low, mid, high = renyiEfficiency(probabilities=unigram_distribution.values(), alpha=2.5)
            wprint(low, mid, high)

            g.add(LOW_KEY,  p, low)
            g.add(MID_KEY,  p, mid)
            g.add(HIGH_KEY, p, high)

    g.commitWithArgs(LineGraph.ArgsGlobal(
        # y_lims=(0.35,0.65),
        x_label=r"BPE dropout probability $p$",
        x_tickspacing=0.1,
        y_label="Rényi efficiency bounds",
        legend_position="lower right"
    ), LineGraph.ArgsPerLine())

    dropouts, lows = g.data[LOW_KEY]
    idx_argmax = np.argmax(lows)
    return dropouts[idx_argmax], lows[idx_argmax]


########################################################################################################################


def main_alphas():
    # Get tokeniser
    kudo = Factory_KudoPiece(kbest=64, alpha=0).buildTokeniser()

    # Get corpus
    _, _, validation_corpus = loadCorpus(CORPUS_ID)

    # Get grid
    # equally_spaced_points = np.linspace(0.05, 0.5, 19)  # Alternatively use 10 instead of 19. The resulting alphas have equally spaced intersections with 1-x, i.e. the curves are equally "spread out".
    # equally_spaced_points = np.linspace(0.05, 0.95, 37)
    equally_spaced_points = np.linspace(0.05, 0.85, 17)  # Powers ranging from 0.02 to 10, since RE seems to increase at least until 1.0.
    alphas = np.log(1-equally_spaced_points)/np.log(equally_spaced_points)

    # Call search
    print("Best alpha and its efficiency:", searchKudoAlpha(
        kudo,
        NamedIterable(validation_corpus, name=validation_corpus.info.dataset_name).map(lambda example: example["text"]),
        alpha_grid=alphas
    ))


def main_dropout():
    # Get corpus
    _, _, validation_corpus = loadCorpus(CORPUS_ID)

    # Get grid
    ps = np.linspace(0,1,21)

    # Call search
    print("Best dropout and its efficiency:", searchDropout(
        NamedIterable(validation_corpus, name=validation_corpus.info.dataset_name).map(lambda example: example["text"]),
        dropout_grid=ps
    ))


def main_temperature(bpe_not_ulm: bool):
    """
    This experiment turns out to be inconclusive at best and pointing to a lower RE for higher temperature at worst,
    which makes a lot of sense since you are decreasing segmentation entropy. You can't fine-tune temperature with RE.
    """
    # Get tokeniser
    if bpe_not_ulm:
        switch = Factory_SwitchyGrampa_BPE(dropout=0.0, l_min=2).buildTokeniser()
    else:
        switch = Factory_SwitchyGrampa_ULM(kbest=1, smoothing_power=1.0, l_min=2).buildTokeniser()
    grampa = switch.subtokenisers[1]
    assert isinstance(grampa, GRaMPa)

    # Get corpus
    _, _, validation_corpus = loadCorpus(CORPUS_ID)

    # Get grid (the original grid was made such that the x^{alpha} curves intersected 1-x at equally spaced points, but the Renyi efficiency basically stayed constant)
    # equally_spaced_points = np.linspace(0.05, 0.5, 19)  # Alternatively use 10 instead of 19.
    # powers = np.log(1-equally_spaced_points)/np.log(equally_spaced_points)
    # temperatures = 1/powers
    temperatures = np.linspace(1.0, 2.0, 21)  # We know that inside this range, the segmentation distribution goes from uniform to skewing massively. Temperatures after 2 are probably irrelevant.

    # Call search
    print("Best temperature and its efficiency:", searchTemperatures(
        grampa,
        NamedIterable(validation_corpus, name=("bpe" if bpe_not_ulm else "kudo") + "_" + validation_corpus.info.dataset_name).map(lambda example: example["text"]),
        temperature_grid=temperatures
    ))


def intrinsicsVersusTemperature_word(word: str, unconstrained: bool=True):
    tk = Factory_GRaMPa(minimal_length=1, vocab_file=KudoPiece32ki_SlimPajama3M()).buildTokeniser()
    renorm = tk.renormalisation
    assert isinstance(renorm, PowerNormalisation)

    # Infinite domain so that we can measure the effect of temperature in a histogram with LOC ordering without weird shit.
    tk.enableInfiniteDomain(unconstrained)

    # for temperature in [1.0, 1.025, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
    # for temperature in [1.0]:
    for temperature in [1.0, 1.025, 1.05, 1.1, 1.15, 1.2]:
    # for temperature in [2.0, 3.0, 4.0, 5.0]:
    # for temperature in [5.0]:
        renorm.resetTemperature(temperature)
        visualiseSingleWordSegmentationDistribution(
            tokenisers=[tk],
            word=word,
            samples=500_000,  # TODO: Recommended for official graphs is 500_000
            segmentation_histogram_max_bins=2**9,
            do_bitbased_ordering=False
        )


def intrinsicsVersusTemperature_corpus_withSubHistograms(corpus: NamedIterable[str], unconstrained: bool=True):
    tk = Factory_GRaMPa(minimal_length=2, vocab_file=KudoPiece32ki_SlimPajama3M()).buildTokeniser()
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


def intrinsicsVersusTemperature_corpus(tk: GRaMPa, word: str, corpus: NamedIterable[str]):
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


def main_multiplex(bpe_not_ulm: bool, temperature: float=1.0):
    # Get tokeniser
    if bpe_not_ulm:
        switch = Factory_SwitchyGrampa_BPE(dropout=0.0, temperature=temperature, l_min=2).buildTokeniser()
    else:
        switch = Factory_SwitchyGrampa_ULM(kbest=1, smoothing_power=1.0, temperature=temperature, l_min=2).buildTokeniser()

    # Get corpus
    _, _, validation_corpus = loadCorpus(CORPUS_ID)

    # Get grid
    grid = np.linspace(0.0, 1.0, 21)

    # Call search
    print("Best multiplex p and its efficiency:", searchMultiplexP(
        switch,
        NamedIterable(validation_corpus, name=validation_corpus.info.dataset_name).map(lambda example: example["text"]),
        probability_grid=grid
    ))


def main_compareBPE():
    """
    Compare stochastic BPE with deterministic BPE, specifically how much of the stochastic tokenisations match the deterministic ones.
    """
    g = LineGraph("diffrate_BPE", caching=CacheMode.WRITE_ONLY)

    # Get corpus
    _, _, validation_corpus = loadCorpus(CORPUS_ID)
    iterable = NamedIterable(validation_corpus, name=validation_corpus.info.dataset_name).map(lambda example: example["text"])

    # Get metric
    metric = ExactMatches(texts=iterable, n_repeats=3, global_preprocessor=TraditionalPreprocessor())

    # Get tokenisers
    deterministic = Factory_BPE(dropout=0.0).buildTokeniser()
    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        wprint(f"Comparing to BPE p={p}...")
        stochastic = Factory_BPE(dropout=p).buildTokeniser()
        ratio, _, _ = metric.compare(deterministic, stochastic)
        g.add(f"$|V| = {deterministic.getVocabSize()}$", p, 1-ratio)

    g.commitWithArgs(
        LineGraph.ArgsGlobal(
            x_label="Dropout rate $p$",
            x_tickspacing=0.1,

            y_label="Word regularisation rate vs. classic BPE",
            y_tickspacing=0.1
        ),
        LineGraph.ArgsPerLine()
    )


def main_compareULM():
    """
    Compare stochastic ULM with deterministic ULM, specifically how much of the stochastic tokenisations match the deterministic ones.
    """
    g = LineGraph("diffrate_ULM", caching=CacheMode.IF_MISSING)

    if g.needs_computation:
        # Get corpus
        _, _, validation_corpus = loadCorpus(CORPUS_ID)
        iterable = NamedIterable(validation_corpus, name=validation_corpus.info.dataset_name).map(lambda example: example["text"])

        # Get metric
        metric = ExactMatches(texts=iterable, n_repeats=3, global_preprocessor=TraditionalPreprocessor())

        # Get tokenisers
        deterministic = Factory_KudoPiece(kbest=1, alpha=1.0).buildTokeniser()
        for a in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0]:
            wprint(f"Comparing to ULM a={a}...")
            stochastic = Factory_KudoPiece(kbest=64, alpha=a).buildTokeniser()
            ratio, _, _ = metric.compare(deterministic, stochastic)
            g.add(f"$|V| = {deterministic.getVocabSize()}$", a, 1-ratio)

    g.commitWithArgs(
        LineGraph.ArgsGlobal(
            x_label=r"Normalisation power $\alpha$",
            x_tickspacing=0.2,

            y_label="Word regularisation rate vs. argmax ULM",
            y_tickspacing=0.1,
            legend_position="upper right"
        ),
        LineGraph.ArgsPerLine()
    )


def main_compareChosenBPEandULM():
    from scripts.constants import BPEDROPOUT_P, ULM_K, ULM_ALPHA

    # Get corpus
    _, _, validation_corpus = loadCorpus(CORPUS_ID, validation_size=1000)
    iterable = NamedIterable(validation_corpus, name=validation_corpus.info.dataset_name).map(lambda example: example["text"])

    # Get metric
    metric = ExactMatches(texts=iterable, n_repeats=3, global_preprocessor=TraditionalPreprocessor())

    # Get tokenisers
    p = BPEDROPOUT_P  # 0.1 is the value with maximal Renyi efficiency, and also recommended by the dropout paper based on BLEU.
    deterministic = Factory_BPE(dropout=0.0).buildTokeniser()
    stochastic    = Factory_BPE(dropout=p).buildTokeniser()
    ratio, _, _ = metric.compare(deterministic, stochastic)
    print(f"BPE vs BPE-dropout({p}):", ratio)

    a = ULM_ALPHA  # 0.15 has RR of 50%. Alternatively, 0.3 is the inflection point of Renyi efficiency, used by Cognetta, and between Kudo's recommended 0.2 and 0.5.
    deterministic = Factory_KudoPiece(kbest=1, alpha=1.0).buildTokeniser()
    stochastic    = Factory_KudoPiece(kbest=ULM_K, alpha=a).buildTokeniser()
    ratio, _, _ = metric.compare(deterministic, stochastic)
    print(f"ULM(k=1) vs ULM(k=64,a={a}):", ratio)


########################################################################################################################


if __name__ == "__main__":
    if IS_NOT_LINUX:
        # main_compareBPE()
        # main_compareULM()
        intrinsicsVersusTemperature_word("antidisestablishmentarianism", unconstrained=True)
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--experiment_multiplex",   default=False, action="store_true")
        parser.add_argument("--experiment_temperature", default=False, action="store_true")
        parser.add_argument("--experiment_alpha",       default=False, action="store_true")
        parser.add_argument("--experiment_dropout",     default=False, action="store_true")

        # Arguments that only hold for multiplexing
        parser.add_argument("--bpe_vocab", default=False, action="store_true")
        parser.add_argument("--fixed_temp", default=1.0, type=float)

        args = parser.parse_args()
        if args.experiment_multiplex:
            main_multiplex(bpe_not_ulm=args.bpe_vocab, temperature=args.fixed_temp)
        elif args.experiment_temperature:
            main_temperature(bpe_not_ulm=args.bpe_vocab)
        elif args.experiment_alpha:
            main_alphas()
        elif args.experiment_dropout:
            main_dropout()
        else:
            raise RuntimeWarning("No hyperparameter experiment was selected.")
