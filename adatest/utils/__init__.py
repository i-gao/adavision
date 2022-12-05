import re
import torch
import urllib
import urllib.parse
import io
import shap
import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import statistics
import json
import warnings

def parse_test_type(test_type):
    part_names = ["text1", "value1", "text2", "value2", "text3", "value3", "text4"]
    parts = re.split(r"(\{\}|\[\])", test_type)
    part_values = ["" for _ in range(7)]
    for i, part in enumerate(parts):
        part_values[i] = part
    return {name: value for name, value in zip(part_names, part_values)}


# https://codereview.stackexchange.com/questions/253198/improved-isinstance-for-ipython
def isinstance_ipython(obj, ref_class):
    def _class_name(obj):
        name = getattr(obj, "__qualname__", getattr(obj, "__name__", ""))
        return (getattr(obj, "__module__", "") + "." + name).lstrip(".")

    return isinstance(obj, ref_class) or _class_name(type(obj)) == _class_name(
        ref_class
    )

@lru_cache(maxsize=1000) # one issue: this will cache None for any images that cause errors...maybe this is preferable?
def _download_image(url, download_timeout=2):
    import PIL
    urllib_request = urllib.request.Request(
        url, data=None,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
        },
    )
    try:
        with urllib.request.urlopen(urllib_request, timeout=download_timeout) as r:
            img_stream = io.BytesIO(r.read())
        return PIL.Image.open(img_stream).convert('RGB')
    except Exception as e:
        return None

# _images_cache = {}
_missing_img_placeholder = _download_image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Black_colour.jpg/180px-Black_colour.jpg",
    download_timeout=20
)
# assert _missing_img_placeholder is not None, "Placeholder image URL in adatest/utils/__init__.py is broken!"

def get_image(urls, download_timeout=10, remove_failures=False):
    """Gets PIL images from URLs, keeping a cache of recently used images."""
    if urls is None: return []
    if type(urls) != list: urls = [urls]

    # download remaining and put in cache
    _download_wrapper = lambda url: _download_image(url.replace("__IMAGE=", ""), download_timeout=download_timeout)
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(_download_wrapper, url) for url in urls] 
    out = [f.result() for f in futures]
    
    # return final list by pulling from cache
    if remove_failures:
        return [
            o for o in out if o is not None
        ]
    else:
        return [
            o if o is not None else _missing_img_placeholder
            for o in out
        ]

def is_subtopic(topic, candidate):
    # Returns true if candidate is a subtopic of topic
    # Both arguments are strings, which look like UNIX paths
    # Return is boolean
    # Note: returns True for self
    topic, candidate = str(topic), str(candidate)
    if len(topic)==len(candidate):
        return topic == candidate
    else:
        return candidate.startswith(topic) and candidate[len(topic)]=='/'

class Stopwatch:
    """
    Context manager for timing a block of code
    Source: https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
    Usage:
        with Stopwatch() as sw:
            # do stuff
        print(sw.time)
    """
    def __enter__(self):
        # torch.cuda.synchronize()
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        # torch.cuda.synchronize()
        self.time = perf_counter() - self.time

def slerp(x, y, t):
    """Returns the spherical interpolation between vectors x, y at parameter t"""
    assert 0 <= t and t <= 1
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    omega = np.arccos(np.dot(x,y))
    if (omega == 0): return x # x, y are the same
    coefx = np.sin((1-t) * omega) / np.sin(omega)
    coefy = np.sin(t * omega) / np.sin(omega)
    return coefx * x + coefy * y

def convert_float(s):
    if s == "":
        return np.nan
    try:
        f = float(s)
    except ValueError:
        f = np.nan
    return f

def safe_json_load(input):
    if isinstance(input, float): # catch NaN's
        return {}
    else:
        return json.loads(input)

def safe_mode(l):
    """ This just silences the error from a double mode from python <= 3.7.
    """
    try:
        return statistics.mode(l)
    except:
        return l[0]

def safe_mean(l):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(l)
    if np.isnan(mean):
        return 0
    else:
        return mean

def score_max(s, nan_val=-1e3):
    """return the maximum of a string a|b|c or a list"""
    if s == "" or s is None:
        return nan_val
    elif isinstance(s, str):
        return np.max([convert_float(v) for v in s.split("|")])
    elif np.isnan(s):
        return nan_val
    else:
        return np.max(s)

def sanitize_topic_name(s, replace_slashes=False):
    """
    Convert a string into a URI encoded topic. (does not enforce opening slash)
    Topic names can't have newlines, trailing spaces, backslashes, or double spaces in their names and must be URI encoded.
    NOTE: if replace_slashes is True, this function will behave incorrectly on inputs that represent the full path of a topic,
    i.e., one with parent/child slashes.
    """
    if type(s) != str: return ""
    s = urllib.parse.unquote(s)
    s = s.replace("\\", "").replace("\n", " ").replace("  ", " ").strip()
    if replace_slashes: 
        s = s.replace("/", " or ")
    return urllib.parse.quote(s)

def pretty_topic_name(s, truncate_to_deepest_subtopic=True):
    """
    Convert a (URI-encoded) topic name of arbitrary depth into a pretty string.
    If truncate_to_deepest_subtopic, we only keep the last subtopic 

    Examples:
    - "/cat%20on%20table" -> "cat on table"
    - "/cat/brown%20cat/small%20brown%20cat"
        -> "small brown cat" if truncate_to_deepest_subtopic
        -> "small brown cat brown cat cat" otherwise
    """
    s = urllib.parse.unquote(s)
    s = s.split("/")[1:]
    if truncate_to_deepest_subtopic:
        return s[:-1]
    else:
        return ' '.join(s[::-1])

def mostly_substring(substr, str):
    """Returns true if more than 1/2 of the words in substr are a part of str"""
    substr = substr.lower().split(" ")
    mask = [s in str.lower() for s in substr]
    return np.count_nonzero(mask) > 0.5 * len(substr)