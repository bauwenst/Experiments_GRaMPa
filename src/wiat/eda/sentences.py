"""
What is the distribution of the amount of tokens produced across sentences of a corpus?
And how does it compare to the maximum input length of a model?
"""
from typing import Iterable
from tktkt.interfaces.tokeniser import Tokeniser


def tokenCounts(tokeniser: Tokeniser, corpus: Iterable[str]) -> Iterable[int]:
    for sentence in corpus:
        yield len(tokeniser.prepareAndTokenise(sentence))
