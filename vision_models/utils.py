import torch
import numpy as np
import requests
import re

from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from urllib.parse import urlparse
from PIL import Image
from scipy.spatial.distance import cdist

def _download_img(url, download_timeout):
    try: 
        bytes = requests.get(url, stream=True, timeout=download_timeout).content
        data = BytesIO(bytes)
        img = Image.open(data).convert('RGB')
        return img
    except Exception as e: 
        return None

def download_img(urls, download_timeout, remove_failures=True):
    """submit parallel download requests"""
    if type(urls) != list: urls = [urls]
    _download_wrapper = lambda u: _download_img(u, download_timeout)
    with Stopwatch() as DT: 
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_download_wrapper, u) for u in urls]
        out = [f.result() for f in futures]
        if remove_failures: out = list(filter(lambda r: r is not None, out))
    print(f"Downloaded images in {DT.time}s.")
    return out

class Stopwatch:
    """
    Context manager for timing a block of code
    Source: https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
    """
    def __enter__(self):
        torch.cuda.synchronize()
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        torch.cuda.synchronize()
        self.time = perf_counter() - self.time

def convert(x, out="array"):
    """Convert tensor / list / array x into a (detached) tensor / list / array"""
    try:
        if type(x) == list and len(x) and not np.isscalar(x[0]):
            return [convert(i, out=out) for i in x]
        if type(x) == dict:
            return {k: convert(v, out=out) for k,v in x.items()}

        if type(x) == list and (not len(x) or np.isscalar(x[0])):
            return convert_scalar_list(x, out)
        elif torch.is_tensor(x): 
            return convert_tensor(x, out)
        elif isinstance(x, np.ndarray):
            return convert_array(x, out)
        elif np.isscalar(x):
            return convert_scalar(x, out)
        else:
            raise ValueError("Invalid type.")
    except:
        return x

def convert_tensor(tensor, out="array"):
    detached_tensor = tensor.detach().clone().cpu()
    if out == "array":
        return detached_tensor.cpu().numpy()
    elif out == "list":
        return detached_tensor.tolist()
    elif out == "tensor":
        return detached_tensor

def convert_array(arr, out="array"):
    if out == "array":
        return arr
    elif out == "list":
        return arr.tolist()
    elif out == "tensor":
        return torch.Tensor(arr)

def convert_scalar(x, out="array"):
    if out == "array":
        return np.array(x)
    elif out == "list":
        return [x]
    elif out == "tensor":
        return torch.Tensor(x)

def convert_scalar_list(lst, out="array"):
    if out == "array":
        return np.array(lst)
    elif out == "list":
        return lst
    elif out == "tensor":
        return torch.Tensor(lst)

def is_img_path(s):
    """Returns if s is a valid path to an image file"""
    try:
        Image.open(s).verify()
        return True
    except:
        return False

def is_url(s):
    """Returns if s is a web URL"""
    try:
        result = urlparse(s)
        return all([result.scheme, result.netloc])
    except:
        return False

def convert_img_to_bytes(img):
    """Convert PIL image to byte array"""
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='jpeg')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def getattr_recursive(obj, att): 
    i = att.find('.')
    if i < 0:
        return getattr(obj, att)
    else:
        return getattr_recursive(
            getattr(obj, att[:i]),
            att[i+1:]
        )

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

RESERVED_CHARS = ": / ? # [ ] @ ! $ & ' ( ) * + ; =".split(' ')
def sanitize_strings(strings):
    """Remove reserved URI characters from all strings in a list"""
    def sanitize_string(s):
        s = s.split(",")[0]
        for c in RESERVED_CHARS: s = s.replace(c, "")
        return s
    return [sanitize_string(s) for s in strings]
