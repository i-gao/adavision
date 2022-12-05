from ._test_tree import TestTree
from ._test_tree_browser import TestTreeBrowser
from ._scorer import Scorer, ClassifierScorer, DetectionScorer, CaptioningScorer
from ._server import serve
from .embedders import _embed as embed
from .embedders import CLIPEmbedding
from . import generators
from ._prompt_builder import PromptBuilder

__version__ = '0.3.4'

default_generators = {
    "tests": generators.CLIPRetriever(),
    "topics": generators.OpenAI(model="text-davinci-002", temperature=0.8, top_p=1),
}
active_embedders = {
    "adatest.embedders.CLIPEmbedding(ViT-L/14)": CLIPEmbedding()
}