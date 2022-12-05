import clip
import torch
from .utils import convert
from scipy.spatial.distance import cdist

class CLIPClassifier(torch.nn.Module):
    """0-shot CLIP classifier"""
    def __init__(self, model_name, output_names=None, prompt=lambda y: f"an image of a {y}"):
        if output_names is None: 
            raise ValueError("0-shot classification using CLIP requires output_names.")

        super().__init__()
        self.clip_model, self.clip_preprocess = clip.load(model_name, jit=True)
        self.output_names = output_names
        self.prompt = prompt
        self._classstring_embs = None
    
    def _embed_classstrings(self):
        classstrings = [self.prompt(y) for y in self.output_names]
        self._classstring_embs = self.embed_txt(classstrings)

    def set_classnames(self, output_names):
        self.output_names = output_names
        self._embed_classstrings()

    def set_prompt(self, prompt):
        self.prompt = prompt
        self._embed_classstrings()

    def forward(self, x):
        """
        Forward pass an image or list of images
        """
        if self._classstring_embs is None:
            self._embed_classstrings()
        
        image_embs = self.embed_img(x)
        sims = 1-cdist(image_embs, self._classstring_embs, metric='cosine')
        return sims

    def embed_img(self, img):
        """Returns the normalized CLIP image embedding given tensors"""
        with torch.no_grad():
            img = img.to(self.device) # redundant w/ Model.forward()
            img_emb = self.clip_model.encode_image(img)
            img_emb /= img_emb.norm(dim=-1, keepdim=True) 
        return convert(img_emb).astype("float32")

    def embed_txt(self, text):
        """Returns the normalized CLIP text embedding given text"""
        if type(text) != list: text = [text]
        if text == []: return np.array([])
        with torch.no_grad():
            text_emb = self.clip_model.encode_text(clip.tokenize(text, truncate=True).to(self.device))
            text_emb /= text_emb.norm(dim=-1, keepdim=True)
        return convert(text_emb).astype("float32")