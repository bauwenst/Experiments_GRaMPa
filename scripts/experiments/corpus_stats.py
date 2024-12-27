from scripts.preamble import *
from wiat.visualisation.sentences import *

from tktkt.util.iterables import take, streamProgress
from tktkt.util.types import NamedIterable
from tktkt.preparation.instances import TraditionalPretokeniser, IdentityMapper, Preprocessor

from scripts.experiments.tokenisers_training import loadCorpus
from scripts.experiments.tokenisers_instances import Factory_SwitchyGrampa_BPE, Factory_KudoPiece, Factory_BPE, TokeniserWithFiniteTypeDomain

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
        iterable=streamProgress(take(CORPUS_SIZE, (example["text"] for example in train)), known_size=CORPUS_SIZE),
        name='/'.join(CORPUS_ID).replace('/', '-') + f"-{CORPUS_SIZE}"
    )

    visualiseCounts(corpus, tk, prep)


if __name__ == "__main__":
    from scripts.constants import *
    # switch = createTokeniser_SwitchyGrampa_BPE(dropout=0.0, t=1.0, l=1)
    # main(switch.subtokenisers[0])
    # main(switch.subtokenisers[1])

    # switch = createTokeniser_SwitchyGrampa_BPE(t=1.0, l=2)
    # main(switch.subtokenisers[1])
    # switch = createTokeniser_SwitchyGrampa_BPE(t=5.0, l=2)
    # main(switch.subtokenisers[1])
    # switch = createTokeniser_SwitchyGrampa_BPE(t=-1.0, l=2)
    # main(switch.subtokenisers[1])
    # switch = createTokeniser_SwitchyGrampa_BPE(t=10.0, l=2)
    # main(switch.subtokenisers[1])
    # switch = createTokeniser_SwitchyGrampa_BPE(t=-0.1, l=2)
    # main(switch.subtokenisers[1])
    bpe = Build_English_BPE(dropout=BPEDROPOUT_P).buildTokeniser()
    main(bpe)

    # kudo = Build_English_Kudo(kbest=ULM_K, alpha=0.1).buildTokeniser()
    # main(kudo)
    # kudo = Build_English_Kudo(kbest=ULM_K, alpha=0.5).buildTokeniser()
    # main(kudo)
    # kudo = Build_English_Kudo(kbest=ULM_K, alpha=1.0).buildTokeniser()
    # main(kudo)
