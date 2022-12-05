from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from .utils import convert, is_url

class AzureCognitiveCaptioner(torch.nn.Module):
    """
    Azure Cognitive Services Vision detection model.

    Set up: see https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration
        - Visit https://portal.azure.com/#create/Microsoft.CognitiveServicesComputerVision to create a Computer Vision resource.
        - Within your new resource, go to "Keys and Endpoint" under "Resource Management."
        - Save the key and endpoint in a file, where the key goes on the first line, and the endpoint is on the second line. The default path is `~/.azure_key`.
    """
    def __init__(self, min_confidence=0.0, api_key_path="~/.azure_key"):
        with open(api_key_path, 'r') as f: 
            API_KEY = f.readline().replace('\n', '')
            ENDPOINT = f.readline().replace('\n', '')
        
        self.client = ComputerVisionClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))
        self.min_confidence = min_confidence

        # azure really wants us to give inputs as URLs, rather than PIL images
        # need to set this flag to signal to the wrapper Model class to not convert URLs to PIL
        self.input_type = "URL"

        # a bunch of hacks to make this look like a torch module
        self._modules, self._parameters, self._buffers = {}, {}, {}
        self._backward_hooks, self._forward_hooks, self._forward_pre_hooks = None, None, None

    def forward(self, x):
        """Expects inputs to be urls."""
        if type(x) == list: assert is_url(x[0])
        else: assert is_url(x); x = [x]

        # submit parallel requests
        # note: sometimes Azure throws errors if URLs are invalid, and we want to signal that this error happened
        # by writing the error in the caption and setting the confidence to 0
        def _client_wrapper(url):
            try: return self.client.analyze_image(url, ["description"])
            except Exception as e: print(e); return None
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_client_wrapper, xp) for xp in x]
        out = [f.result() for f in futures]

        return [self.result_to_caption(o) for o in out]

    def result_to_caption(self, res):
        """Converts Azure ImageAnalysis Result object to dict with keys 'captions' and 'scores'.
        Also handles failed calls."""
        if res is None:
            captions = ['[call to model failed]']
            scores = [0]
        else:
            res = res.description.captions
            captions, scores = [], []
            for r in res:
                if r.confidence >= self.min_confidence:
                    captions.append(r.text)
                    scores.append(r.confidence)
        return {
            'captions': convert(captions),
            'scores': convert(scores),
        }

class AzureCognitiveDetector(torch.nn.Module):
    """
    Azure Cognitive Services Vision detection model.

    Set up: see https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration
        - Visit https://portal.azure.com/#create/Microsoft.CognitiveServicesComputerVision to create a Computer Vision resource.
        - Within your new resource, go to "Keys and Endpoint" under "Resource Management."
        - Save the key and endpoint in a file, where the key goes on the first line, and the endpoint is on the second line. The default path is `~/.azure_key`.
    """
    def __init__(self, min_confidence=0.85, api_key_path="~/.azure_key"):
        with open(api_key_path, 'r') as f: 
            API_KEY = f.readline().replace('\n', '')
            ENDPOINT = f.readline().replace('\n', '')
        
        self.client = ComputerVisionClient(ep, CognitiveServicesCredentials(key))
        self.min_confidence = min_confidence
        self.output_names = None

        # azure really wants us to give inputs as URLs, rather than PIL images
        # need to set this flag to signal to the wrapper Model class to not convert URLs to PIL
        self.input_type = "URL"

        # a bunch of hacks to make this look like a torch module
        self._modules, self._parameters, self._buffers = {}, {}, {}
        self._backward_hooks, self._forward_hooks, self._forward_pre_hooks = None, None, None

    def forward(self, x):
        """Expects inputs to be urls."""
        if type(x) == list: assert is_url(x[0])
        else: assert is_url(x); x = [x]

        # submit parallel requests
        _client_wrapper = lambda url: self.client.analyze_image(
            url, ["objects"]
        )
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_client_wrapper, xp) for xp in x]
        out = [f.result() for f in futures]

        return [self.result_to_detection(o) for o in out]

    def result_to_detection(self, res):
        """Converts Amazon Results dict object to torchvision detection dict object"""
        assert self.output_names is not None
        res = res.objects
        labels, scores, boxes = [], [], []
        for r in res:
            if r.confidence >= self.min_confidence:
                labels.append(self.output_names.index(r.object_property))
                scores.append(r.confidence)
                boxes.append(self.xywh_to_xyxy(r.rectangle))
        return {
            'labels': convert(labels, out='tensor').int(),
            'boxes': convert(boxes),
            'scores': convert(scores, out='tensor'),
        }

    def xywh_to_xyxy(self, rectangle):
        """Converts Amazon Instance dict object to a 4-tuple of (xmin, ymin, xmax, ymax) format"""
        w = rectangle.w
        h = rectangle.h
        left = rectangle.x
        top = rectangle.y
        return [left * shape[0], top * shape[1], (left + w) * shape[0], (top + h) * shape[1]]