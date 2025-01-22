from scripts.preamble import *

import time
from typing import Iterable, List, Tuple
from datasets import load_dataset, Dataset
from math import sqrt

from tktkt.factories.preprocessing import TruncateAndNormalise, TraditionalPretokeniser
from tktkt.interfaces import Preprocessor
from tktkt.interfaces.tokeniser import Tokeniser
from tktkt.interfaces.factories import TokeniserFactory
from tktkt.evaluation.speed import secondsPerTokenisation
from tktkt.factories.deserialisation import KudoPiece32ki_SlimPajama3M
from tktkt.models.random.pathmarkov import GRaMPa
from tktkt.models.random.rejectionsampling import RandomVocabSegmentation_RejectionSampling_UniformGraph as Cognetta
from tktkt.models.identity.segmentation import IdentityTokeniser
from tktkt.preparation.mappers import IdentityMapper
from tktkt.util.printing import wprint
from tktkt.util.types import NamedIterable
from tktkt.util.iterables import take, streamProgress
from tktkt.wrappers.multiplexing import StochasticTokeniserSwitch

from fiject import Table, LineGraph, CacheMode, ColumnStyle

from scripts.visualisation.table_abstract import SortableRowKeys, FormattedKeys

vocab = KudoPiece32ki_SlimPajama3M(specials=[])
pretoken_generator = Preprocessor(TruncateAndNormalise(1_000_000), IdentityMapper(), TraditionalPretokeniser())


def getTokeniserFactories() -> List[Tuple[str,TokeniserFactory]]:
    from scripts.experiments.lineages import LINEAGES, LineageRootNode

    factories = []
    for l in LINEAGES:
        n = l._node_tree
        assert isinstance(n, LineageRootNode)
        factories.append((l.name,n._tokeniser))
    return factories


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


from scripts.visualisation.table_finetuning import GRaMPaFinetuningParser, GRaMPaRowKey, VOCABS
class KeyFormatter(GRaMPaFinetuningParser):
    def __init__(self):
        super().__init__(dict(),dict())

    def _to_sortkey_row(self, key: GRaMPaRowKey) -> SortableRowKeys:
        return super()._to_sortkey_row(key) if key.infer != "BPE" else (VOCABS.index(key.vocab), -1, 0, 0)

    def _format_row(self, key: GRaMPaRowKey) -> FormattedKeys:
        formatted = super()._format_row(key)
        if formatted[1] == "BPE-dropout":
            formatted[1] = "BPE"
        return formatted


def intrinsicMetrics(corpus: NamedIterable[str], n_examples: int):
    """
    TODO: Maybe you want to do this for a lemma corpus and a real-text corpus.
          One gives more information about the tokeniser, the other about what models see in practice.
    """
    parser = KeyFormatter()

    table = Table(f"intrinsics-{corpus.name}-{n_examples}", caching=CacheMode.IF_MISSING, overwriting=True)
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
        for text in streamProgress(take(n_examples, corpus), known_size=n_examples):
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

            table.set(tk_stats.l.mean(), row, [r"$\ell$", col_mean])
            table.set(tk_stats.l.std(),  row, [r"$\ell$", col_std])

            table.set(tk_stats.R.mean(), row, ["$R$", col_mean])
            table.set(tk_stats.R.std(),  row, ["$R$", col_std])

            table.set(tk_stats.S.mean(), row, [r"$\mathcal S$", col_mean])
            table.set(tk_stats.S.std(),  row, [r"$\mathcal S$", col_std])

    table.commit(
        default_column_style=ColumnStyle(do_bold_maximum=True, do_bold_minimum=True),
        borders_between_rows_of_level=[0,1,2],
        borders_between_columns_of_level=[0]
    )


def slowdownGraph(corpus: NamedIterable[str], n_examples: int,
                  tokenisers: List[Tokeniser]):
    """
    Tests how Cognetta slows down over a corpus when you gradually increase the word length limit.
    Beyond 30, we know it basically just hangs forever.

    On an i7, you can count about 25 minutes per limit per 1000 examples.
    6 limits implies 150 minutes (2.5 hours).
    """
    LIMITS = [5, 10, 15, 20, 25, 30]
    metrics = [[BCF() for _ in LIMITS] for _ in tokenisers]

    graph = LineGraph(f"slowdown-{corpus.name}-{n_examples}", caching=CacheMode.IF_MISSING)
    if graph.needs_computation:
        # Test that the tokenisers work
        for t in tokenisers:
            t.prepareAndTokenise("This is a test sentence.")

        # Per-tokeniser
        for l_idx, limit in enumerate(LIMITS):
            wprint("Limit:", limit)
            for text in streamProgress(take(n_examples, corpus), known_size=n_examples):
                for pretoken in pretoken_generator.do(text):
                    pretoken_limited = pretoken[:limit]
                    for t_idx, t in enumerate(tokenisers):
                        start = time.perf_counter()
                        t.prepareAndTokenise(pretoken_limited)
                        end = time.perf_counter()

                        metrics[t_idx][l_idx].add(end-start)

    # TODO: Now put it in a graph that supports highlighting standard deviation.
    # from fiject import StochasticLineGraph


def speedTest(name: str, tokeniser: Tokeniser, corpus: Iterable[str], n_examples: int):
    wprint(f"Speedtesting {name}...")
    avg, std = secondsPerTokenisation(
        tokeniser,
        streamProgress(take(n_examples, corpus), known_size=n_examples)
    )
    print("\t Average [s/tk]:", avg)
    print("\tStd. dev [s/tk]:", std)


if __name__ == "__main__":
    # L = [2, 4, 4, 4, 5, 5, 7, 9]
    # b = BCF()
    # for x in L:
    #     b.add(x)
    # print(b.amount(), b.mean(), b.std())  # Should be 8, m=5, s=2.138 (Wikipedia example)

    print("Loading dataset...")
    corpus = NamedIterable(load_dataset("allenai/c4", "en", streaming=True)["train"], "C4").map(lambda e: e["text"])
    N = 500

    intrinsicMetrics(corpus, N)

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
