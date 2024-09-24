from tst.preamble import *

from typing import Tuple

from wiat.visualisation.tokenisers import *

from tktkt.preparation.instances import CommonsensePreprocessor, RobertaSpaceMarker
from tktkt.preparation.boundaries import BoundaryMarker
from tktkt.models.bpe.base import ClassicBPE
from tktkt.models.random.pathmarkov import RandomVocabSegmentation_GreedyMarkov, PowerNormalisation

from bpe_knockout import morphologyGenerator, KnockoutDataConfiguration, setupEnglish

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def getBaseHuggingFaceTokeniser() -> Tuple[PreTrainedTokenizerBase, BoundaryMarker]:
    return AutoTokenizer.from_pretrained("roberta-base"), RobertaSpaceMarker


def main_BPE_corpus():
    tk = ClassicBPE.fromHuggingFace(getBaseHuggingFaceTokeniser()[0], for_words=True)

    with KnockoutDataConfiguration(setupEnglish()):
        visualiseCharsVersusTokensRelationships(
            tokeniser=tk,
            raw_words=(o.word for o in morphologyGenerator(verbose=True)),
            n_samples_per_word=1
        )


def main_GRaMPa_word(unconstrained: bool=True):
    hftk, marker = getBaseHuggingFaceTokeniser()

    # for temperature in [1.0, 1.025, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
    for temperature in [1.0, 1.025, 1.05, 1.1, 1.15, 1.2, 1.25]:
        tk = RandomVocabSegmentation_GreedyMarkov(
            preprocessor=CommonsensePreprocessor(marker),
            vocab=hftk.get_vocab(),

            minimal_token_length=2,
            decode_backwards=True,
            probabilities_to_probabilities=PowerNormalisation(temperature=temperature)
        )

        # Infinite domain so that we can measure the effect of temperature in a histogram with LOC ordering without weird shit.
        tk.enableInfiniteDomain(unconstrained)

        word = "antidisestablishmentarianism"
        visualiseSingleWordSegmentationDistribution(
            tokeniser=tk,
            word=word,
            samples=100_000,  # TODO: Recommended for official graphs is 500_000
            segmentation_histogram_max_bins=2**9,
            do_bitbased_ordering=False
        )


def main_GRaMPa_corpus(unconstrained: bool = True):
    hftk, marker = getBaseHuggingFaceTokeniser()

    tk = RandomVocabSegmentation_GreedyMarkov(
        preprocessor=CommonsensePreprocessor(marker),
        vocab=hftk.get_vocab(),

        minimal_token_length=2,
        decode_backwards=False,
        probabilities_to_probabilities=PowerNormalisation(temperature=1.0)
    )

    # Infinite domain so that we can measure the effect of temperature in a histogram with LOC ordering without weird shit.
    tk.enableInfiniteDomain(unconstrained)

    with KnockoutDataConfiguration(setupEnglish()):
        visualiseCharsVersusTokensRelationships(
            tokeniser=tk,
            raw_words=(o.word for o in morphologyGenerator()),
            n_samples_per_word=100
        )


if __name__ == "__main__":
    # main_GRaMPa_word(unconstrained=False)
    main_BPE_corpus()
