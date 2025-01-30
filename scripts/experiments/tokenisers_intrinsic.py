"""
Tests to do with fertility and speed, metrics inherent to the tokeniser itself rather than downstream tasks,
WITHOUT changing any hyperparameters.
"""
from scripts.preamble import *
from scripts.experiments.tokenisers_training import loadCorpus, IS_NOT_LINUX, CORPUS_ID
from scripts.experiments.lineages import getTokeniserFactories
from scripts.visualisation.table_abstract import SortableRowKeys, FormattedKeys
from scripts.visualisation.table_instances import GRaMPaFinetuningParser, GRaMPaRowKey, VOCABS

import time
from typing import Iterable, List, Tuple
from math import sqrt

from tktkt.interfaces import Preprocessor
from tktkt.interfaces.tokeniser import Tokeniser
from tktkt.wrappers.multiplexing import StochasticTokeniserSwitch
from tktkt.evaluation.speed import secondsPerTokenisation
from tktkt.visualisation.charts.token_distributions import visualiseCharsVersusTokensRelationships
from tktkt.factories.preprocessing import TruncateAndNormalise, TraditionalPretokeniser, IdentityMapper
from tktkt.factories.tokenisers import Factory_BPE
from tktkt.util.printing import wprint
from tktkt.util.types import NamedIterable
from tktkt.util.iterables import take, streamProgress, mapExtend

from fiject import Table, LineGraph, CacheMode, ColumnStyle, MultiHistogram, FIJECT_DEFAULTS, \
    StreamingVariableGranularityHistogram, BinSpec, BinOverlapMode, VariableGranularityHistogram, StreamingStochasticLineGraph

pretoken_generator = Preprocessor(TruncateAndNormalise(1_000_000), IdentityMapper(), TraditionalPretokeniser())
def pretokenIterableFromCorpus(line_iterable: NamedIterable[str]) -> NamedIterable[str]:
    """Turns a named iterable of texts into a named iterable of words."""
    return line_iterable.flatmap(pretoken_generator.do)


def loadCorpusAsNamedIterable() -> NamedIterable[str]:
    _, _, validation = loadCorpus(CORPUS_ID)
    return NamedIterable(validation, "C4" if IS_NOT_LINUX else "SlimPajama").map(lambda e: e["text"])


class BCF:
    """
    1-dimensional BIRCH clustering feature (N,LS,SS). Also known as a streamable mean and streamable standard deviation.
    https://arxiv.org/pdf/2006.12881
    """

    def __init__(self):
        self._n  = 0
        self._w  = 0
        self._ls = 0
        self._ss = 0

    def add(self, x: float, weight: float=1.0):
        self._n  += 1
        self._w  += weight
        self._ls += weight*x
        self._ss += weight*x**2

    def amount(self) -> int:
        return self._n

    def weight(self) -> int:
        return self._w

    def mean(self) -> float:
        return self._ls / self.weight()

    def std(self, ddof: int=1) -> float:
        n = self.amount()
        W = self.weight()
        return sqrt( 1/(n-ddof) * n/W * (self._ss - W*self.mean()**2) )


class IntrinsicMetricsKeyFormatter(GRaMPaFinetuningParser):
    def __init__(self):
        super().__init__(dict(),dict())

    def _to_sortkey_row(self, key: GRaMPaRowKey) -> SortableRowKeys:
        return super()._to_sortkey_row(key) if key.infer != "BPE" else (VOCABS.index(key.vocab), -1, 0, 0)

    def _format_row(self, key: GRaMPaRowKey) -> FormattedKeys:
        formatted = super()._format_row(key)
        if formatted[1] == "BPE-dropout":
            formatted[1] = "BPE"
        return formatted


########################################################################################################################


def intrinsicMetrics(line_iterable: NamedIterable[str], n_examples: int):
    """
    TODO: Maybe you want to do this for a lemma corpus and a real-text corpus.
          One gives more information about the tokeniser, the other about what models see in practice.
    """
    parser = IntrinsicMetricsKeyFormatter()

    table = Table(f"intrinsics-{line_iterable.name}-{n_examples}", caching=CacheMode.IF_MISSING, overwriting=True)
    if table.needs_computation:
        # Get the tokenisers.
        keys       = []
        tokenisers = []
        vocabs_with_deterministic_tk = set()
        for n,f in getTokeniserFactories():
            key = parser._extractRowKey({"Name": n})

            tk = f.buildTokeniser()
            if isinstance(tk, StochasticTokeniserSwitch):  # => Also add the separate parts.
                for tk in tk.subtokenisers:
                    if "GRaMPa" in tk.getName():  # ---> The GRaMPa part.
                        keys.append(key)
                        tokenisers.append(tk)
                    elif key.vocab not in vocabs_with_deterministic_tk:  # ---> The non-GRaMPa part for a vocab we don't have the tokeniser of already.
                        vocabs_with_deterministic_tk.add(key.vocab)

                        if key.vocab == "BPE":
                            keys.append(GRaMPaRowKey(key.vocab, "BPE-dropout", None, 0.0))
                        elif key.vocab == "ULM":
                            keys.append(GRaMPaRowKey(key.vocab, key.vocab, 1, 1.0))
                        else:
                            raise NotImplementedError
                        tokenisers.append(tk)
            else:
                keys.append(key)
                tokenisers.append(tk)

        # Initialise the metrics per tokeniser.
        class IntrinsicMetrics:
            def __init__(self):
                self.l = BCF()
                self.m = BCF()
                self.R = BCF()
                self.S = BCF()

        metrics = [IntrinsicMetrics() for _ in tokenisers]
        keys_tokenisers_metrics = list(zip(keys,tokenisers,metrics))

        ###
        for k,t,m in keys_tokenisers_metrics:
            print(t, parser._to_sortkey_row(k), parser._format_row(k), sep="\n\t")
        ###

        # Iterate over all pretokens in the dataset, tokenising them and collecing statistics.
        for text in streamProgress(take(n_examples, line_iterable), known_size=n_examples):
            for pretoken in pretoken_generator.do(text):
                for _, tk, tk_stats in keys_tokenisers_metrics:
                    tokens = tk.prepareAndTokenise(pretoken)

                    s = sum(map(len,tokens))  # Preprocessor-specific.
                    m = len(tokens)
                    for token in tokens:
                        l = len(token)
                        tk_stats.l.add(l)
                    R = s/m
                    S = (m-1)/(s-1)
                    tk_stats.m.add(m)
                    tk_stats.R.add(R)
                    tk_stats.S.add(S)

        for key, tk, tk_stats in sorted(keys_tokenisers_metrics, key=lambda t: parser._to_sortkey_row(t[0])):
            row = parser._format_row(key)
            col_mean = "mean"
            col_std  = "std"

            table.set(tk_stats.m.mean(), row, ["$m$", col_mean])
            table.set(tk_stats.m.std(),  row, ["$m$", col_std])

            table.set(tk_stats.S.mean(), row, [r"$\mathcal S$", col_mean])
            table.set(tk_stats.S.std(),  row, [r"$\mathcal S$", col_std])

            table.set(tk_stats.l.mean(), row, [r"$\ell$", col_mean])
            table.set(tk_stats.l.std(),  row, [r"$\ell$", col_std])

            table.set(tk_stats.R.mean(), row, ["$R$", col_mean])
            table.set(tk_stats.R.std(),  row, ["$R$", col_std])

    table.commit(
        default_column_style=ColumnStyle(do_bold_maximum=True, do_bold_minimum=True),
        borders_between_rows_of_level=[0,1,2],
        borders_between_columns_of_level=[0]
    )


def intrinsicMetricsHistograms(tokenisers: List[Tuple[Tokeniser,str]], pretoken_iterable: NamedIterable[str], exclude_words_over_length: int=60,
                               do_boxplots: bool=False):
    FIJECT_DEFAULTS.GLOBAL_STEM_PREFIX = pretoken_iterable.name

    graph_lengths = MultiHistogram("token-length")
    graph_amounts = MultiHistogram("amounts")
    graph_cpt     = MultiHistogram("cpt")
    # graph_segmentality = MultiHistogram("segmentality")
    graph_segmentality = StreamingVariableGranularityHistogram("segmentality", BinSpec.closedFromAmount(0,1,amount=30))

    for raw_word in pretoken_iterable:
        if len(raw_word) > exclude_words_over_length or len(raw_word) < 2:
            continue

        for tokeniser,name in tokenisers:
            tokens = tokeniser.prepareAndTokenise(raw_word)

            n_chars = 0
            for l in map(len, tokens):
                graph_lengths.add(name, l)
                n_chars += l
            n_tokens = len(tokens)
            graph_amounts.add(name, n_tokens)
            graph_cpt    .add(name, n_chars/n_tokens)
            # graph_segmentality.add(name, (n_tokens-1)/(n_chars-1))
            graph_segmentality.add(n_tokens-1, n_chars, class_name=name)

    if do_boxplots:
        graph_lengths.commitWithArgs_boxplot(MultiHistogram.ArgsGlobal_BoxPlot())
        graph_amounts.commitWithArgs_boxplot(MultiHistogram.ArgsGlobal_BoxPlot())
        graph_cpt    .commitWithArgs_boxplot(MultiHistogram.ArgsGlobal_BoxPlot())
        # graph_segmentality.commitWithArgs_boxplot(MultiHistogram.ArgsGlobal_BoxPlot())
    else:
        graph_lengths.commitWithArgs_histplot(MultiHistogram.ArgsGlobal(
            x_label="Token length",
            center_ticks=True,

            y_label="Fraction of tokens",
            relative_counts=True
        ))
        graph_amounts.commitWithArgs_histplot(MultiHistogram.ArgsGlobal(
            x_label="Amount of tokens",
            center_ticks=True,

            y_label="Fraction of words",
            relative_counts=True
        ))
        # graph_cpt.commitWithArgs_histplot(MultiHistogram.ArgsGlobal(
        #     x_label="Characters-per-token ratio",
        #     binwidth=0.025,
        #
        #     y_label="Fraction of words",
        #     relative_counts=True
        # ))
        # graph_segmentality.commitWithArgs_histplot(MultiHistogram.ArgsGlobal(
        #     x_label="Segmentality",
        #     binwidth=0.025,
        #
        #     y_label="Spread across words",
        #     relative_counts=True
        # ))
        graph_segmentality.commit(StreamingVariableGranularityHistogram.ArgsGlobal(
            x_label="Segmentality",
            x_tickspacing=0.1,
            histo_overlapping=BinOverlapMode.SIDE_BY_SIDE,

            y_label="Spread across words",
            relative_counts=True
        ))
        print(graph_segmentality.getSummaries())

    FIJECT_DEFAULTS.GLOBAL_STEM_PREFIX = ""


def segmentalityHistogram(tk: Tokeniser, pretoken_iterable: NamedIterable[str], n_samples: int=100):
    segmentality = VariableGranularityHistogram(f"{tk.getName()}_{pretoken_iterable.name}_{n_samples}_segmentality", caching=CacheMode.WRITE_ONLY)

    if segmentality.needs_computation:
        for word in pretoken_iterable:
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


def corpusHistograms(word_corpus: NamedIterable[str]):
    tk = Factory_BPE().buildTokeniser()
    visualiseCharsVersusTokensRelationships(
        tokeniser=tk,
        raw_words=word_corpus,
        n_samples_per_word=1
    )


########################################################################################################################


def slowdownGraph(line_iterable: NamedIterable[str], n_examples: int,
                  tokenisers: List[Tokeniser]):
    """
    Tests how Cognetta slows down over a corpus when you gradually increase the word length limit.
    Beyond 30, we know it basically just hangs forever.

    On an i7, you can count about 25 minutes per limit per 1000 examples.
    6 limits implies 150 minutes (2.5 hours).
    """
    LIMITS = [5, 10, 15, 20, 25, 30]

    graph = StreamingStochasticLineGraph(f"slowdown-{line_iterable.name}-{n_examples}", caching=CacheMode.IF_MISSING)
    if graph.needs_computation:
        # Test that the tokenisers work
        for t in tokenisers:
            t.prepareAndTokenise("This is a test sentence.")

        # Per-tokeniser
        for limit in LIMITS:
            wprint("Limit:", limit)
            for text in streamProgress(take(n_examples, line_iterable), known_size=n_examples):
                for pretoken in pretoken_generator.do(text):
                    pretoken_limited = pretoken[:limit]
                    for tk in tokenisers:
                        start = time.perf_counter()
                        tk.prepareAndTokenise(pretoken_limited)
                        end = time.perf_counter()

                        graph.addSample(tk.getName(), limit, end-start)

    graph.commit(
        StreamingStochasticLineGraph.ArgsGlobal(
            x_label="Pretoken truncation limit [chars]",
            y_label="Time per pretoken [s]",

            uncertainty_opacity=0.25
        ),
        StreamingStochasticLineGraph.ArgsPerLine()
    )


def speedTest(name: str, tokeniser: Tokeniser, corpus: Iterable[str], n_examples: int):
    wprint(f"Speedtesting {name}...")
    avg, std = secondsPerTokenisation(
        tokeniser,
        streamProgress(take(n_examples, corpus), known_size=n_examples)
    )
    print("\t Average [s/tk]:", avg)
    print("\tStd. dev [s/tk]:", std)


########################################################################################################################


def tst_bcf():
    L = [2, 4, 4, 4, 5, 5, 7, 9]
    b = BCF()
    for x in L:
        b.add(x)
    print(b.amount(), b.mean(), b.std())  # Should be 8, m=5, s=2.138 (Wikipedia example)


def main1():
    """
    Generate table with inference metrics across the project corpus.
    """
    intrinsicMetrics(loadCorpusAsNamedIterable(), n_examples=75 if IS_NOT_LINUX else 20_000)


def main2():
    """
    Generate histogram of vocabulary type lengths.
    """
    from scripts.experiments.lineages import BPE32ki_SlimPajama3M, KudoPiece32ki_SlimPajama3M_New
    from tktkt.visualisation.charts.token_distributions import visualiseTypes
    visualiseTypes([BPE32ki_SlimPajama3M(specials=[]), KudoPiece32ki_SlimPajama3M_New(specials=[])],
                   ["BPE", "ULM"])


def main3():
    """
    For each of the vocabularies, generate hypothetical vocabulary fertility stats across the project corpus.
    """
    from scripts.experiments.lineages import BPE32ki_SlimPajama3M, KudoPiece32ki_SlimPajama3M_New
    from tktkt.models.identity.segmentation import IdentityTokeniserWithVocab
    from tktkt.evaluation.fertility import getVocabStats

    deserialisers = [BPE32ki_SlimPajama3M(specials=[]), KudoPiece32ki_SlimPajama3M_New(specials=[])]
    pretokens = pretokenIterableFromCorpus(loadCorpusAsNamedIterable())
    for d in deserialisers:
        print(d.__class__.__name__)
        print("\t", getVocabStats(prep_and_vocab=IdentityTokeniserWithVocab(preprocessor=d.preprocessorEffective(), vocab=d.buildVocabulary()),
                                  raw_words=pretokens,
                                  do_log_segmentations=True))


def main4():
    from tktkt.evaluation.entropy import segmentationDistributionFromWord, analyseSegmentationDistribution
    N = 100_000
    # TODO: You should average this across many words. To average entropy, you may want to turn it into efficiency.
    print(tk.getName())
    print("\t", analyseSegmentationDistribution(segmentationDistributionFromWord(tk, "antidisestablishmentarianism", N), N, deterministic_segmentation=None))


if __name__ == "__main__":
    main3()

    # from tktkt.util.iterables import keepFirst
    # from tktkt.util.types import NamedIterable
    #
    # from bpe_knockout import morphologyGenerator, KnockoutDataConfiguration, setupEnglish
    #
    # class DummyIterableThatJustMakesGenerators(Iterable):
    #     def __iter__(self):
    #         with KnockoutDataConfiguration(setupEnglish()):
    #             return keepFirst(o.word for o in morphologyGenerator(verbose=True))
    # word_corpus = NamedIterable(DummyIterableThatJustMakesGenerators(), name="celex-en-lemmata")

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

    # plot_fertilities(tokenisers, raw_words=word_corpus)

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

    # from tktkt.models.random.pathmarkov import GRaMPa
    # from tktkt.models.random.rejectionsampling import RandomVocabSegmentation_RejectionSampling_UniformGraph as Cognetta
    # from tktkt.models.identity.segmentation import IdentityTokeniser
    # from tktkt.factories.deserialisation import KudoPiece32ki_SlimPajama3M
    # vocab = KudoPiece32ki_SlimPajama3M(specials=[])
    # slowdownGraph(
    #     corpus,
    #     N,
    #     [
    #         GRaMPa(
    #             preprocessor=vocab.preprocessorEffective(),
    #             vocab=vocab.buildVocabulary(),
    #             decode_backwards=False
    #         ),
    #         Cognetta(
    #             preprocessor=vocab.preprocessorEffective(),
    #             vocab=vocab.buildVocabulary()
    #         )
    #     ]
    # )

    ################################

    # Rough estimations:
    #   Cognetta: 1.5 seconds/it
    #   GRaMPa:   20 it/second
    # Improvement: 30x (but it's much more on the intractable examples).
    # speedTest("Preprocessor overhead", IdentityTokeniser(
    #     preprocessor=preprocessor
    # ), corpus, N)
    # speedTest("GRaMPa", GRaMPa(
    #     preprocessor=preprocessor,
    #     vocab=vocab.buildVocabulary(),
    #     decode_backwards=False
    # ), corpus, N)
    # speedTest("Cognetta", Cognetta(
    #     preprocessor=preprocessor,
    #     vocab=vocab.buildVocabulary()
    # ), corpus, N)
