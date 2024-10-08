"""
Rather than give a model the result of applying simple preprocessing + tokenisation (e.g. with ULM),
insert the reverse of each pretoken before tokenising and see if the model can figure this out.
"""
from lamoto.augmenting.augment_dataset import Task, PerturbWords

from tktkt.interfaces import Preprocessor
from tktkt.preparation.huggingface import HuggingFaceNormaliser, tn
from tktkt.preparation.mappers import IdentityMapper, PseudoByteMapping
from tktkt.preparation.splitters import *
from tktkt.preparation.instances import RobertaSpaceMarker
from tktkt.preparation.perturbers import *
from tktkt.models.huggingface.wrapper import HuggingFaceTokeniser

from transformers import AutoTokenizer


class TyposLevenshtein1(PerturbWords):
    def __init__(self, task: Task, text_field_name: str, p: float=0.10):
        sampler = ConstantSampler(n=1)
        super().__init__(task, text_field_name=text_field_name, mapping=ParallelPerturber(
            p=p,
            perturbations=[
                Substitute(0, sampler=sampler),
                Insert(0, sampler=sampler),
                Pop(0, sampler=sampler)
            ]
        ))


########################################################################################################################


class CommonsenseWithReverse(PretokeniserSequence):
    def __init__(self, marker: BoundaryMarker, do_reverse: bool):
        super().__init__(
            [
                PunctuationPretokeniser(HyphenMode.EXCLUDED, protect_apostrophes_without_spaces=True),
                WhitespacePretokeniser(destructive=True)
            ] +\
            do_reverse * [
                InsertReverse(),  # Insert the reverse BEFORE byte mapping and BEFORE the word boundary, because the reverse of byte mappings doesn't exist and there are no types in the vocab with the word boundary on the opposite end.
            ] +\
            [
                MapperAsPretokeniser(PseudoByteMapping()),
                EnglishApostrophes(do_nt=True),
                AddWordBoundary(marker),
                IsolateDigits(),
                PunctuationPretokeniser(HyphenMode.ONLY),
            ]
        )


reversing_pretokeniser = CommonsenseWithReverse(RobertaSpaceMarker, do_reverse=True)
preprocessor = Preprocessor(
    uninvertible_mapping=HuggingFaceNormaliser(tn.NFKC()),
    invertible_mapping=IdentityMapper(),
    splitter=reversing_pretokeniser
)

def tst_reversingprep():
    """
    A test to see what the model would roughly see by using the reversing preprocessor.

    Hmmmm, kind of fragmented, and also some other preprocessor stuff causes some words to correspond to more than
    one pretoken.
    """
    s = "This is a sentence like any other sentence brother (but it has 69420 \"strange\" quirks). That's it."
    print(preprocessor.do(s))

    # tk = Builder_English_BPE_native().buildTokeniser()
    tk = HuggingFaceTokeniser(wrapped_tokeniser=AutoTokenizer.from_pretrained("roberta-base"))
    tk.preprocessor = preprocessor
    print(tk.prepareAndTokenise(s))

    pretokens = preprocessor.do(s)
    for forward, backward in zip(pretokens[::2], pretokens[1::2]):
        print(forward, backward)
        print(tk.tokenise(forward), tk.tokenise(backward))


if __name__ == "__main__":
    tst_reversingprep()
