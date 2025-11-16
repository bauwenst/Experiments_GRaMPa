from scripts.preamble import *

from tktkt.util.iterables import take, streamProgress
from tktkt.util.types import NamedIterable
from tktkt.factories.preprocessing import TraditionalPretokeniser, IdentityMapper, Preprocessor
from tktkt.interfaces.tokeniser import TokeniserWithFiniteTypeDomain
from tktkt.factories.tokenisers import Factory_SwitchyGrampa_BPE, Factory_KudoPiece, Factory_BPE
from tktkt.visualisation.charts.token_distributions import visualiseCharsVersusTokensRelationships

from scripts.experiments.tokenisers_training import loadCorpus

CORPUS_ID   = ("allenai/c4", "en")
# CORPUS_SIZE = 2_000_000
# CORPUS_ID = ("cerebras/SlimPajama-627B",)
CORPUS_SIZE = 2_000


def main(tk: TokeniserWithFiniteTypeDomain):
    # Load word isolator
    prep = Preprocessor(IdentityMapper(), IdentityMapper(), TraditionalPretokeniser())

    # Load corpus
    _, train, _ = loadCorpus(CORPUS_ID, train_size=CORPUS_SIZE)
    corpus = NamedIterable(
        iterable=train,
        name='/'.join(CORPUS_ID).replace('/', '-') + f"-{CORPUS_SIZE}"
    ).map(lambda example: example["text"]).wrap(lambda it: take(CORPUS_SIZE, it)).tqdm().flatmap(prep.do)

    visualiseCharsVersusTokensRelationships(
        tokenisers=[tk],
        raw_words=corpus,
        n_samples_per_word=1,
        do_progressbar=False
    )


if __name__ == "__main__":
    from scripts.constants import *
    # switch = Factory_SwitchyGrampa_BPE(dropout=0.0, t=1.0, l=1).buildTokeniser()
    # main(switch.subtokenisers[0])
    # main(switch.subtokenisers[1])

    # switch = Factory_SwitchyGrampa_BPE(t=1.0, l=2).buildTokeniser()
    # main(switch.subtokenisers[1])
    # switch = Factory_SwitchyGrampa_BPE(t=5.0, l=2).buildTokeniser()
    # main(switch.subtokenisers[1])
    # switch = Factory_SwitchyGrampa_BPE(t=-1.0, l=2).buildTokeniser()
    # main(switch.subtokenisers[1])
    # switch = Factory_SwitchyGrampa_BPE(t=10.0, l=2).buildTokeniser()
    # main(switch.subtokenisers[1])
    # switch = Factory_SwitchyGrampa_BPE(t=-0.1, l=2).buildTokeniser()
    # main(switch.subtokenisers[1])
    bpe = Factory_BPE(dropout=BPEDROPOUT_P).buildTokeniser()
    main(bpe)

    # kudo = Build_English_Kudo(kbest=ULM_K, alpha=0.1).buildTokeniser()
    # main(kudo)
    # kudo = Build_English_Kudo(kbest=ULM_K, alpha=0.5).buildTokeniser()
    # main(kudo)
    # kudo = Build_English_Kudo(kbest=ULM_K, alpha=1.0).buildTokeniser()
    # main(kudo)
