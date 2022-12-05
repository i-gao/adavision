import numpy as np
import adatest
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
import appdirs
import torch
import diskcache
_embedding_memory_cache = {}
_embedding_file_cache = diskcache.Cache(appdirs.user_cache_dir("adatest") + "/embeddings.diskcache")

def _embed(strings: list, normalize=True, embedder_name=None):
    """
    Embeds string and image objects and caches embeddings.
    Main function of this file. Exported in __init__.py as adatest.embed()
    Example downstream uses:
        - to enforce prompt diversity in a PromptBuilder
        - to fit TopicLabeling and TopicMembership models 
        - to embed queries for retrieval generators
    Args:
        - strings: the items to be embedded. Expected to be of type [str], where str may be text or an image URL, formatted as "__IMAGE=URL"
        - normalize: whether to normalize embeddings before storing in cache
        - embedder_name: name of embedder to use. If None, defaults to the first compatible (image/text) embedder in adatest.active_embedders.
            Names of supported embedders (defined as classes in this file):
                - adatest.embedders.TransformersTextEmbedding(MODEL_NAME)
                - adatest.embedders.OpenAITextEmbedding(MODEL_NAME)
                - adatest.embedders.CLIPEmbedding(MODEL_NAME)
    """
    if strings is None: return []
    assert type(strings) == list

    text_prefix = _get_text_embedder(embedder_name).name
    image_prefix = _get_image_embedder(embedder_name).name

    # check cache for any objs that have cached embeddings
    prefixed_strings = [image_prefix + s if s.startswith("__IMAGE=") else text_prefix + s for s in strings]
    urls_to_embed, text_to_embed = [], []
    for i, prefixed_s in enumerate(prefixed_strings):       
        if prefixed_s not in _embedding_memory_cache:
            if prefixed_s not in _embedding_file_cache:
                s = strings[i]
                if s.startswith("__IMAGE="):
                    urls_to_embed.append(s)
                else:
                    text_to_embed.append(s)
                _embedding_memory_cache[prefixed_s] = None # so we don't embed the same string twice
            else:
                _embedding_memory_cache[prefixed_s] = _embedding_file_cache[prefixed_s]
    
    # embed the new text strings
    if len(text_to_embed) > 0:
        new_embeds = _get_text_embedder(embedder_name)(text_to_embed)
        for i, s in enumerate(text_to_embed):
            prefixed_s = text_prefix + s
            if normalize:
                _embedding_memory_cache[prefixed_s] = new_embeds[i] / np.linalg.norm(new_embeds[i])
            else:
                _embedding_memory_cache[prefixed_s] = new_embeds[i]
            _embedding_file_cache[prefixed_s] = _embedding_memory_cache[prefixed_s]

    # embed the new image urls
    if len(urls_to_embed) > 0:
        with adatest.utils.Stopwatch() as DT: 
            images = adatest.utils.get_image(urls_to_embed)
        print(f"embedder.py: downloaded {len(urls_to_embed)} images in {DT.time}s.")
        with adatest.utils.Stopwatch() as ET:
            new_embeds = _get_image_embedder(embedder_name)(images)
        print(f"embedder.py: embedded {len(images)} images in {ET.time}s.")
        for i, s in enumerate(urls_to_embed):
            prefixed_s = image_prefix + s
            if normalize:
                _embedding_memory_cache[prefixed_s] = new_embeds[i] / np.linalg.norm(new_embeds[i])
            else:
                _embedding_memory_cache[prefixed_s] = new_embeds[i]
            _embedding_file_cache[prefixed_s] = _embedding_memory_cache[prefixed_s]
    
    # finally, pull from cache and return
    result = [
        _embedding_memory_cache[prefixed_s] for prefixed_s in prefixed_strings
    ]
    return [e.squeeze() for e in result]

def _get_text_embedder(name):
    """ Get the text embedding model."""
    # if specific embedder requested
    if name is not None:
        if name in adatest.active_embedders: 
            return adatest.active_embedders[name]
        else:
            # parse name
            try:
                constructor_name = name.split("(")[0].split(".")[-1]
                model_name = name.split("(")[1].split(")")[0]
                embedder = globals()[constructor_name](model=model_name)
                adatest.active_embedders[name] = embedder
                return embedder
            except:
                print(f"Invalid embedder {name}. Using first compatible embedder in adatest.active_embedders instead.")
                pass
    
    # use the first available text embedder
    for e in adatest.active_embedders:
        if adatest.active_embedders[e].embeds_text:
            return adatest.active_embedders[e]
    
    # add a text embedder; defaults to Transformers
    embedder = TransformersTextEmbedding()
    adatest.active_embedders[embedder.name] = embedder
    return embedder

def _get_image_embedder(name):
    """ Get the image embedding model."""
    # if specific embedder requested
    if name is not None:
        if name in adatest.active_embedders: 
            return adatest.active_embedders[name]
        else:
            # parse name
            try:
                constructor_name = name.split("(")[0].split(".")[-1]
                model_name = name.split("(")[1].split(")")[0]
                embedder = globals()[constructor_name](model=model_name)
                adatest.active_embedders[name] = embedder
                return embedder
            except:
                print(f"Invalid embedder {name}. Using first compatible embedder in adatest.active_embedders instead.")
                pass
    
    # use the first available image embedder
    embedder = None
    for e in adatest.active_embedders:
        if adatest.active_embedders[e].embeds_image:
            return adatest.active_embedders[e]

    # add a image embedder; defaults to CLIP
    embedder = CLIPEmbedding()
    adatest.active_embedders[embedder.name] = embedder
    return embedder

#####

def cos_sim(a, b):
    """ Cosine distance between two vectors.
    """
    return normalize(a, axis=1) @ normalize(b, axis=1).T


def unique(qs, threshold=0.999, return_index=False):
    # compute cossine sims between all pairs of images
    qs = np.vstack(qs)
    sims = 1-cdist(qs, qs, metric='cosine')
    
    # identify duplicate pairs by cossine sim >= threshold
    i, j = np.where(sims >= threshold)
    mask = (j > i)
    
    # get indices of vectors to keep
    i, j = i[mask], j[mask]
    unique = set(np.arange(len(sims))) - set(j)
    unique = np.array(list(unique))
    
    if return_index:
        return qs[unique], unique
    else:
        return qs[unique]

#####

class TransformersTextEmbedding():
    def __init__(self, model="sentence-transformers/stsb-roberta-base-v2"):
        import transformers

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model)
        self.model = transformers.AutoModel.from_pretrained(model).to(self.device)
        self.model_name = model
        self.name = "adatest.embedders.TransformersTextEmbedding(" + self.model_name + "):"
        self.embeds_text = True
        self.embeds_image = False

    def __call__(self, strings):
        encoded_input = self.tokenizer(strings, padding=True, truncation=True, return_tensors='pt').to(self.device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform mean pooling
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        embeds = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return embeds.cpu().numpy().astype('float32')

class OpenAITextEmbedding():
    def __init__(self, model="text-similarity-babbage-001", api_key=None, replace_newlines=True):
        import openai
        self.model = model
        if api_key is not None:
            openai.api_key = api_key
        self.replace_newlines = replace_newlines
        self.model_name = model
        self.name = "adatest.embedders.OpenAITextEmbedding(" + self.model_name + "):"
        self.embeds_text = True
        self.embeds_image = False

    def __call__(self, strings):
        import openai

        if len(strings) == 0:
            return np.array([])

        # clean the strings for OpenAI
        cleaned_strings = []
        for s in strings:
            if s == "":
                s = " " # because OpenAI doesn't like empty strings
            elif self.replace_newlines:
                s = s.replace("\n", " ") # OpenAI recommends this for things that are not code
            cleaned_strings.append(s)
        
        # call the OpenAI API to complete the prompts
        response = openai.Embedding.create(
            input=cleaned_strings, model=self.model, user="adatest"
        )

        return np.vstack([e["embedding"] for e in response["data"]])

class CLIPEmbedding():
    def __init__(self, model="ViT-L/14"):
        import clip

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device, jit=True)
        self.clip_tokenize = clip.tokenize
        self.model_name = model
        self.name = "adatest.embedders.CLIPEmbedding(" + self.model_name + "):"
        self.embeds_text = True
        self.embeds_image = True

    def __call__(self, obj):
        if (type(obj) == list and type(obj[0]) == str) or type(obj) == str:
            return self.embed_txt(obj)
        else:
            return self.embed_img(obj)

    def embed_img(self, img):
        """Returns the normalized CLIP image embedding for a list of PIL images"""
        if type(img) != list: img = [img]
        if img == []: return np.array([])
        with torch.no_grad():
            img = torch.cat([self.clip_preprocess(i).unsqueeze(0) for i in img], dim=0).to(self.device)
            img_emb = self.clip_model.encode_image(img)
        return img_emb.cpu().numpy().astype("float32")

    def embed_txt(self, text):
        """Returns the normalized CLIP text embedding for a list of strings"""        
        if type(text) != list: text = [text]
        if text == []: return np.array([])
        with torch.no_grad():
            text_emb = self.clip_model.encode_text(self.clip_tokenize(text, truncate=True).to(self.device))
        return text_emb.cpu().numpy().astype("float32")