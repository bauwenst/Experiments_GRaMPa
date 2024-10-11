from tst.preamble import *
from wiat.visualisation.sentences import *

from tktkt.util.iterables import take, streamProgress
from tktkt.preparation.instances import TraditionalPretokeniser, IdentityMapper, Preprocessor

from tst.experiments.tokenisers_training import loadCorpus
from tst.experiments.tokenisers_instances import createTokeniser_SwitchyGrampa_BPE, Build_English_Kudo, Build_English_BPE

CORPUS_ID   = ("allenai/c4", "en")
# CORPUS_SIZE = 2_000_000
# CORPUS_ID = ("cerebras/SlimPajama-627B",)
CORPUS_SIZE = 2_000


def main(tk: Tokeniser):
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
    bpe = Build_English_BPE(dropout=0.1).buildTokeniser()
    main(bpe)

    # kudo = Build_English_Kudo(kbest=64, alpha=0.1).buildTokeniser()
    # main(kudo)
    # kudo = Build_English_Kudo(kbest=64, alpha=0.5).buildTokeniser()
    # main(kudo)
    # kudo = Build_English_Kudo(kbest=64, alpha=1.0).buildTokeniser()
    # main(kudo)
