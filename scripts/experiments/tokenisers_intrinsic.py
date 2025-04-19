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
from collections import Counter

from tktkt.interfaces import Preprocessor
from tktkt.interfaces.tokeniser import Tokeniser
from tktkt.wrappers.multiplexing import StochasticTokeniserSwitch
from tktkt.evaluation.speed import secondsPerTokenisation
from tktkt.evaluation.fertility import prepareAndCountValidSegmentations
from tktkt.evaluation.entropy import bitKeyFromTokens, normaliseCounter, analyseSegmentationDistribution
from tktkt.visualisation.charts.token_distributions import visualiseCharsVersusTokensRelationships
from tktkt.factories.preprocessing import TruncateAndNormalise, TraditionalPretokeniser, IdentityMapper
from tktkt.factories.tokenisers import Factory_BPE
from tktkt.factories.deserialisation import KudoPiece32ki_SlimPajama3M
from tktkt.util.printing import wprint
from tktkt.util.types import NamedIterable
from tktkt.util.iterables import take, streamProgress

from fiject import Table, CacheMode, ColumnStyle, MultiHistogram, FIJECT_DEFAULTS, \
    StreamingVariableGranularityHistogram, BinSpec, BinOverlapMode, VariableGranularityHistogram, StreamingStochasticLineGraph

pretoken_generator = Preprocessor(TruncateAndNormalise(1_000_000), IdentityMapper(), TraditionalPretokeniser())
def pretokenIterableFromCorpus(line_iterable: NamedIterable[str]) -> NamedIterable[str]:
    """Turns a named iterable of texts into a named iterable of words."""
    return line_iterable.flatmap(pretoken_generator.do)


def loadValidationCorpusAsNamedIterable() -> NamedIterable[str]:
    _, _, validation = loadCorpus(CORPUS_ID)
    return NamedIterable(validation, name=validation.info.dataset_name).map(lambda e: e["text"])


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

    def std(self, ddof: int=1) -> float:  # Only correct when weights represent frequencies.
        W = self.weight()
        return sqrt(max(1/(W-ddof) * (self._ss - W*self.mean()**2), 0))  # The max(..., 0) is just because rounding error sometimes causes the subtraction to be a very small negative number.


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


def intrinsicMetrics(line_iterable: NamedIterable[str], n_examples: int, n_samples_per_pretoken: int=1):
    """
    TODO: Maybe you want to do this for a lemma corpus and a real-text corpus.
          One gives more information about the tokeniser, the other about what models see in practice.
    """
    parser = IntrinsicMetricsKeyFormatter()

    table = Table(f"intrinsics-{line_iterable.name}-{n_examples}×{n_samples_per_pretoken}", caching=CacheMode.IF_MISSING, overwriting=True)
    table2 = Table(f"entropies-{line_iterable.name}-{n_examples}×{n_samples_per_pretoken}", caching=CacheMode.IF_MISSING, overwriting=True)
    if table.needs_computation or table2.needs_computation:
        # Get the tokenisers.
        keys       = []
        tokenisers = []
        is_deterministic = []
        effective_preprocessors = []
        vocabs_with_deterministic_tk = set()
        for n,f in getTokeniserFactories():
            key = parser._extractRowKey({"Name": n})

            tk = f.buildTokeniser()
            if isinstance(tk, StochasticTokeniserSwitch):  # => Also add the separate parts.
                for tk in tk.subtokenisers:
                    if "GRaMPa" in tk.getName():  # ---> The GRaMPa part.
                        keys.append(key)
                        tokenisers.append(tk)
                        is_deterministic.append(False)
                        effective_preprocessors.append(tk.preprocessor)
                    elif key.vocab not in vocabs_with_deterministic_tk:  # ---> The non-GRaMPa part for a vocab we don't have the tokeniser of already.
                        vocabs_with_deterministic_tk.add(key.vocab)

                        if key.vocab == "BPE":
                            keys.append(GRaMPaRowKey(key.vocab, "BPE-dropout", None, 0.0))
                            effective_preprocessors.append(tk.preprocessor)
                        elif key.vocab == "ULM":
                            keys.append(GRaMPaRowKey(key.vocab, key.vocab, 1, 1.0))
                            effective_preprocessors.append(KudoPiece32ki_SlimPajama3M().preprocessorEffective())
                        else:
                            raise NotImplementedError
                        tokenisers.append(tk)
                        is_deterministic.append(True)
            else:
                keys.append(key)
                tokenisers.append(tk)
                is_deterministic.append(False)
                effective_preprocessors.append(tk.preprocessor if key.vocab != "ULM" else KudoPiece32ki_SlimPajama3M().preprocessorEffective())

        # Initialise the metrics per tokeniser.
        class IntrinsicMetrics:
            def __init__(self):
                self.m = BCF()
                self.S = BCF()
                self.l = BCF()
                self.R = BCF()

                self.max_coverage_uniqueness = BCF()
                self.coverage           = BCF()
                self.uniqueness         = BCF()
                self.rr_versus_argmax   = BCF()  # Intuitively, this is always larger than uniqueness. Uniqueness measures for each segmentation if it is different from ALL segmentations so far. RR measures for each segmentation if it is different from one particular segmentation.
                self.eff_all            = BCF()
                self.eff_without_argmax = BCF()

        metrics = [IntrinsicMetrics() for _ in tokenisers]
        keys_tokenisers_preprocessors_determinism_metrics = list(zip(keys,tokenisers,effective_preprocessors,is_deterministic,metrics))

        ###
        print("Tested tokenisers:")
        for k,t,p,d,m in keys_tokenisers_preprocessors_determinism_metrics:
            print(t, d, parser._to_sortkey_row(k), parser._format_row(k), sep="\n\t")
        ###

        # Iterate over all pretokens in the dataset, tokenising them and collecting statistics.
        for text in streamProgress(take(n_examples, line_iterable), known_size=n_examples):
            for word in pretoken_generator.do(text):
                for _, tk, prep, is_det, tk_stats in keys_tokenisers_preprocessors_determinism_metrics:
                    segmentation_distribution = Counter()
                    for _ in range(n_samples_per_pretoken if not is_det else 1):
                        tokens = tk.prepareAndTokenise(word)

                        # Update count metrics
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

                        # Update segmentation distribution
                        segmentation_distribution[bitKeyFromTokens(tokens)] += 1 if not is_det else n_samples_per_pretoken

                    domain_size, _, _ = prepareAndCountValidSegmentations(word, prep, tk.vocab)
                    if domain_size != 1:  # We are not interested in polluting our averages with data about pretokens whose distribution is always the same regardless of the tokeniser. It is pointless to consider these for comparing tokenisers, and additionally, a distribution with only one value could be seen as both perfectly random and perfectly deterministic at the same time, corresponding to Rényi efficiencies of 1 and 0 respectively. The choice of how to resolve this ambiguity is arbitrary and yet has quite a large effect because 0 and 1 both pull the mean and variance towards extremes.
                        distributional_stats = analyseSegmentationDistribution(
                            normaliseCounter(segmentation_distribution),
                            domain_size=domain_size,
                            sample_size=n_samples_per_pretoken
                        )
                        tk_stats.max_coverage_uniqueness .add(distributional_stats.max_coverage_uniqueness)
                        tk_stats.coverage                .add(distributional_stats.coverage)
                        tk_stats.uniqueness              .add(distributional_stats.uniqueness)
                        tk_stats.eff_all           .add(distributional_stats.efficiency_all)
                        tk_stats.eff_without_argmax.add(distributional_stats.efficiency_no_argmax)
                        tk_stats.rr_versus_argmax  .add(distributional_stats.regularisation_rate_argmax)
                        ###
                        # if is_det:
                        #     print(tk, "applied to", prep.do(word))
                        #     print("\t Domain size:", prepareAndCountValidSegmentations(word, prep, tk.vocab)[0])
                        #     print("\tDistribution:", list(segmentation_distribution.values()))
                        #     print("\t   Entropies:", distributional_stats)
                        ###

        for key, tk, _, _, tk_stats in sorted(keys_tokenisers_preprocessors_determinism_metrics, key=lambda t: parser._to_sortkey_row(t[0])):
            row = parser._format_row(key)
            col_mean = "mean"
            col_std  = "std"
            versus_argmax = r"\cancel{\text{m}}"

            # First table
            col = "$m$"
            table.set(tk_stats.m.mean(), row, [col, col_mean])
            table.set(tk_stats.m.std(),  row, [col, col_std])

            col = r"$\mathcal S$"
            table.set(tk_stats.S.mean(), row, [col, col_mean])
            table.set(tk_stats.S.std(),  row, [col, col_std])

            col = r"$\ell$"
            table.set(tk_stats.l.mean(), row, [col, col_mean])
            table.set(tk_stats.l.std(),  row, [col, col_std])

            col = "$R$"
            table.set(tk_stats.R.mean(), row, [col, col_mean])
            table.set(tk_stats.R.std(),  row, [col, col_std])

            # Second table
            # - Highest number is the entropic efficiency without the argmax.
            col = "$H_1^{" + versus_argmax + "}/H_0^{" + versus_argmax + "}$"
            table2.set(tk_stats.eff_without_argmax.mean(), row, [col, col_mean])
            table2.set(tk_stats.eff_without_argmax.std(),  row, [col, col_std])

            # - Then comes the entropic efficiency with argmax.
            col = "$H_1/H_0$"
            table2.set(tk_stats.eff_all.mean(), row, [col, col_mean])
            table2.set(tk_stats.eff_all.std(),  row, [col, col_std])

            # - Then regularisation rate w.r.t. the argmax
            col = r"RR${}^{" + versus_argmax + "}$"
            table2.set(tk_stats.rr_versus_argmax.mean(), row, [col, col_mean])
            table2.set(tk_stats.rr_versus_argmax.std(),  row, [col, col_std])

            # - Then uniqueness/coverage
            col = "$\max(C,U)$"
            table2.set(tk_stats.max_coverage_uniqueness.mean(), row, [col, col_mean])
            table2.set(tk_stats.max_coverage_uniqueness.std(),  row, [col, col_std])

            col = "$C$"
            table2.set(tk_stats.coverage.mean(), row, [col, col_mean])
            table2.set(tk_stats.coverage.std(),  row, [col, col_std])

            col = "$U$"
            table2.set(tk_stats.uniqueness.mean(), row, [col, col_mean])
            table2.set(tk_stats.uniqueness.std(),  row, [col, col_std])

    table.commit(
        default_column_style=ColumnStyle(do_bold_maximum=True, do_bold_minimum=True),
        borders_between_rows_of_level=[0,1,2],
        borders_between_columns_of_level=[0]
    )
    table2.commit(
        default_column_style=ColumnStyle(
            cell_function=lambda x: 100*x
        ),
        borders_between_rows_of_level=[0, 1, 2],
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

    37 minutes for 200x10 examples on my home i7.

    Say we want 100 samples per pretoken, then doing 2000 examples gets us to about (37/60) * (2000*100)/(200*10) = 62 hours.
    Take 10 hours of margin to get 3 days.

    TODO: Note that you want 20k for table 1 but 2k for table 2.
    """
    intrinsicMetrics(
        loadValidationCorpusAsNamedIterable(),
        n_examples=9 if IS_NOT_LINUX else 2_000,
        n_samples_per_pretoken=10 if IS_NOT_LINUX else 100
    )


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
    from tktkt.evaluation.fertility import getVocabStats

    deserialisers = [BPE32ki_SlimPajama3M(specials=[]), KudoPiece32ki_SlimPajama3M_New(specials=[])]
    pretokens = pretokenIterableFromCorpus(loadValidationCorpusAsNamedIterable())
    for d in deserialisers:
        print(d.__class__.__name__)
        print("\t", getVocabStats(effective_preprocessor=d.preprocessorEffective(), vocab=d.buildVocabulary(),
                                  raw_words=pretokens,
                                  do_log_segmentations=True))


def main4():
    # Import tokenisers
    from tktkt.models.identity.segmentation import IdentityTokeniser
    from tktkt.models.random.rejectionsampling import RandomVocabSegmentation_RejectionSampling_UniformGraph as Cognetta
    from tktkt.models.random.pathmarkov import GRaMPa

    # Import vocab
    from tktkt.factories.deserialisation import KudoPiece32ki_SlimPajama3M
    vocab = KudoPiece32ki_SlimPajama3M(specials=[])

    # Test
    slowdownGraph(
        loadValidationCorpusAsNamedIterable(),
        n_examples=1000 if IS_NOT_LINUX else 20_000,
        tokenisers=[
            IdentityTokeniser(
                preprocessor=vocab.preprocessorEffective()
            ),
            GRaMPa(
                preprocessor=vocab.preprocessorEffective(),
                vocab=vocab.buildVocabulary(),
                decode_backwards=False
            ),
            Cognetta(
                preprocessor=vocab.preprocessorEffective(),
                vocab=vocab.buildVocabulary()
            )
        ]
    )


if __name__ == "__main__":
    main1()

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
