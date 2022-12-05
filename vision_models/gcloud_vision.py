from google.cloud.vision_v1 import ImageAnnotatorClient, Feature
from google.cloud.vision_v1 import Image as VisionImage
from PIL import Image
import torch

from .utils import convert_img_to_bytes, convert

class GoogleCloudVisionClassifier(torch.nn.Module):
    """
    Google Cloud Vision API Label Detection model.

    Set up: see https://codelabs.developers.google.com/codelabs/cloud-vision-api-python#1
        - Activate the Google Cloud Vision API for your project
        - Create a service account
        - Save the service account key (JSON) locally and set up an environmental variable GOOGLE_APPLICATION_CREDENTIALS=~/key.json

    Args:
        - mids (list) Google Knowledge Graph entity IDs to use as allowed class predictions
    """
    def __init__(self, mids):
        self.client = ImageAnnotatorClient()
        self.mids = mids

        self._stack_batch = False
        
        # a bunch of hacks to make this look like a torch module
        self._modules, self._parameters, self._buffers = {}, {}, {}
        self._backward_hooks, self._forward_hooks, self._forward_pre_hooks = None, None, None

    def forward(self, x):
        """Expects inputs to be either bytes or images."""
        if type(x) == list: assert type(x[0]) == bytes or type(x[0]) == Image.Image
        else: assert type(x) == bytes or type(x) == Image.Image

        # format requests dict & convert to bytes if image
        if type(x) == list:
            x = [{
                'image': VisionImage(content=xp if type(xp) == bytes else convert_img_to_bytes(xp)),
                'features': [{'type_': Feature.Type.LABEL_DETECTION}]
                } for xp in x]
        else:
            x = [{
                'image': VisionImage(content=x if type(x) == bytes else convert_img_to_bytes(x)),
                'features': [{'type_': Feature.Type.LABEL_DETECTION}]
                }]

        out = self.client.batch_annotate_images(requests=x) # BatchAnnotateImagesResponse        
        return torch.stack([self.result_to_row(r) for r in out.responses], dim=0)

    def result_to_row(self, res):
        """Converts GCloud AnnotateImagesResponse object to tensor of scores per known label"""
        assert self.mids is not None
        res = res.label_annotations
        labels, scores = [], []
        for r in res:
            try: labels.append(self.mids.index(r.mid)) # if this ever throws an error, we will need to manually patch the mid list
            except: 
                print(f"Google Cloud Vision detected a Label with an MID that is not in our MID list: {r.mid}")
                continue # just omit this label for now
            scores.append(r.score)
        
        # build matrix to return
        matrix = torch.zeros(len(self.mids))
        matrix[labels] = torch.tensor(scores)
        return matrix


class GoogleCloudVisionDetector(torch.nn.Module):
    """
    Google Cloud Vision API Label Detection model.

    Set up: see https://codelabs.developers.google.com/codelabs/cloud-vision-api-python#1
        - Activate the Google Cloud Vision API for your project
        - Create a service account
        - Save the service account key (JSON) locally and set up an environmental variable GOOGLE_APPLICATION_CREDENTIALS=~/key.json
    """
    def __init__(self, mids):
        self.client = ImageAnnotatorClient()
        self.mids = mids

        self._stack_batch = False

        # a bunch of hacks to make this look like a torch module
        self._modules, self._parameters, self._buffers = {}, {}, {}
        self._backward_hooks, self._forward_hooks, self._forward_pre_hooks = None, None, None

    def forward(self, x):
        """Expects inputs to be either bytes or images."""
        if type(x) == list: assert type(x[0]) == Image.Image
        else: assert type(x) == Image.Image

        # format requests dict & convert to bytes if image
        if type(x) == list:
            sizes = [xp.size for xp in x]
            x = [{
                'image': VisionImage(content=xp if type(xp) == bytes else convert_img_to_bytes(xp)),
                'features': [{'type_': Feature.Type.OBJECT_LOCALIZATION}]
                } for xp in x]
        else:
            sizes = [x.size]
            x = [{
                'image': VisionImage(content=x if type(x) == bytes else convert_img_to_bytes(x)),
                'features': [{'type_': Feature.Type.OBJECT_LOCALIZATION}]
                }]

        out = self.client.batch_annotate_images(requests=x) # BatchAnnotateImagesResponse        
        return [self.result_to_detection(r, sizes[i]) for i, r in enumerate(out.responses)]

    def result_to_detection(self, res, size):
        """Converts GCloud AnnotateImagesResponse object to torchvision detection dict object"""
        assert self.mids is not None
        res = res.localized_object_annotations
        labels, scores, boxes = [], [], []
        for r in res:
            try: labels.append(self.mids.index(r.mid)) # if this ever throws an error, we will need to manually patch the mid list
            except: 
                print(f"Google Cloud Vision detected a Label with an MID that is not in our MID list: {r.mid}")
                continue # just omit this label for now
            scores.append(r.score)
            boxes.append(self.boundingpoly_to_xyxy(r.bounding_poly, size))
        return {
            'labels': convert(labels, out='tensor').int(),
            'boxes': convert(boxes),
            'scores': convert(scores, out='tensor'),
        }

    def boundingpoly_to_xyxy(self, bp, size):
        """Converts BoundingPoly object to a 4-tuple of (xmin, ymin, xmax, ymax) format"""
        w, h = size
        bp = bp.normalized_vertices
        xs = [v.x for v in bp]
        ys = [v.y for v in bp]
        return [min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h]
