from transformers import OFATokenizer, OFAModel
from transformers.models.ofa.generate import sequence_generator
import torch
from .utils import convert
from scipy.spatial.distance import cdist
import pathlib
from torchvision import transforms
from PIL import Image
import re

file_path = str(pathlib.Path(__file__).parent.resolve())

class OFACaptioner(torch.nn.Module):
    """
    OFA-huge vision + language generative model
    Based on: https://colab.research.google.com/drive/1LLJewY92LXdeug5m_ceMUHdlqrRQwSQJ?usp=sharing#scrollTo=WgLUTdMIuUb_
    """
    def __init__(self, checkpoint_dir=f"{file_path}/OFA-huge-caption", prompt="What does the image describe?", **generator_kwargs):
        super().__init__()
        self.tokenizer = OFATokenizer.from_pretrained(checkpoint_dir)
        self.model = OFAModel.from_pretrained(checkpoint_dir, use_cache=False)
        self.set_prompt(prompt)

        self.generator_kwargs = {'num_beams': 5, 'no_repeat_ngram_size': 3, 'max_length': 15}
        self.generator_kwargs.update(generator_kwargs)

        self._stack_batch = True

    def set_prompt(self, prompt):
        assert type(prompt) == str
        self.prompt_ids = self.tokenizer([prompt], return_tensors="pt").input_ids

    def forward(self, x):
        """
        Forward pass a tensor or a stack of tensors
        Expects ndims=4
        TODO: can we get actual scores from the logprobs?
        """
        gen = self.model.generate(
            input_ids=torch.stack([self.prompt_ids] * len(x)).squeeze(1).squeeze(1).to(x.device),
            patch_images=x,
            **self.generator_kwargs
        )
        results = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        results = [re.sub(r'[^\w\s]', '', r).strip() for r in results]

        # package results
        return [{
            'captions': [r],
            'scores': [1.0],
        } for r in results]

