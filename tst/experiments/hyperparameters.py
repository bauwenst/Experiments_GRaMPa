from tst.preamble import *
from tst.experiments.tokenisers_instances import createTokeniser_SwitchyGrampa_BPE, createTokeniser_SwitchyGrampa_ULM
from tst.experiments.tokenisers_training import loadCorpus, CORPUS_ID

from typing import Iterable, Tuple
import numpy as np

from tktkt.evaluation.entropy import renyiEfficiency, tokenDistributionFromSentences
from tktkt.models.random.pathmarkov import RandomVocabSegmentation_GreedyMarkov, PowerNormalisation
from tktkt.models.kudopiece.segmentation import KudoPieceTokeniser
from tktkt.wrappers.multiplexing import StochasticTokeniserSwitch
from tktkt.util.types import NamedIterable
from tktkt.util.printing import wprint
from tktkt.util.environment import IS_NOT_LINUX

from fiject import LineGraph, CacheMode


LOW_KEY  = r"$H_\alpha/\lceil H_0\rceil$"
MID_KEY  = r"$H_\alpha/H_0$"
HIGH_KEY = r"$\lceil H_\alpha \rceil/H_0$"


def searchTemperatures(markov_tokeniser: RandomVocabSegmentation_GreedyMarkov, corpus: NamedIterable[str], temperature_grid: Iterable[float]) -> Tuple[float,float]:
    normaliser = markov_tokeniser.renormalisation
    assert isinstance(normaliser, PowerNormalisation)
    normaliser.tau = "t"  # Trick to make .getName() not depend on the init setting.
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
    kudo_tokeniser._alpha = "a"  # Trick to make .getName() not depend on the init setting.
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


########################################################################################################################


def main_temperature(bpe_not_ulm: bool):
    # Get tokeniser
    if bpe_not_ulm:
        switch = createTokeniser_SwitchyGrampa_BPE()
    else:
        switch = createTokeniser_SwitchyGrampa_ULM()
    grampa = switch.subtokenisers[1]
    assert isinstance(grampa, RandomVocabSegmentation_GreedyMarkov)

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


def main_multiplex(bpe_not_ulm: bool, temperature: float=1.0):
    # Get tokeniser
    if bpe_not_ulm:
        switch = createTokeniser_SwitchyGrampa_BPE()
    else:
        switch = createTokeniser_SwitchyGrampa_ULM()

    # Get corpus
    _, _, validation_corpus = loadCorpus(CORPUS_ID)

    grampa = switch.subtokenisers[1]
    assert isinstance(grampa, RandomVocabSegmentation_GreedyMarkov)
    renorm = grampa.renormalisation
    assert isinstance(renorm, PowerNormalisation)
    renorm.tau = temperature

    # Get grid
    grid = np.linspace(0.0, 1.0, 21)

    # Call search
    print("Best multiplex p and its efficiency:", searchMultiplexP(
        switch,
        NamedIterable(validation_corpus, name=validation_corpus.info.dataset_name).map(lambda example: example["text"]),
        probability_grid=grid
    ))


def main_alphas():
    # Get tokeniser
    switch = createTokeniser_SwitchyGrampa_ULM()
    kudo = switch.subtokenisers[0]
    assert isinstance(kudo, KudoPieceTokeniser)

    # Get corpus
    _, _, validation_corpus = loadCorpus(CORPUS_ID)

    # Get grid
    equally_spaced_points = np.linspace(0.05, 0.5, 19)  # Alternatively use 10 instead of 19.
    alphas = np.log(1-equally_spaced_points)/np.log(equally_spaced_points)

    # Call search
    print("Best alpha and its efficiency:", searchKudoAlpha(
        kudo,
        NamedIterable(validation_corpus, name=validation_corpus.info.dataset_name).map(lambda example: example["text"]),
        alpha_grid=alphas
    ))


########################################################################################################################


if __name__ == "__main__":
    if IS_NOT_LINUX:
        main_multiplex(bpe_not_ulm=True)
        main_multiplex(bpe_not_ulm=False)
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--experiment_multiplex",   default=False, action="store_true")
        parser.add_argument("--experiment_temperature", default=False, action="store_true")
        parser.add_argument("--experiment_alpha",       default=False, action="store_true")

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
        else:
            raise RuntimeWarning("No hyperparameter experiment was selected.")
