import boto3
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from .utils import convert_img_to_bytes, convert

class AmazonRekognition(torch.nn.Module):
    """
    Amazon Rekognition detection model.

    Set up: see https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration
        - Visit AWS and create an IAM account.
        - In ~/.aws/credentials, save your AWS access key and AWS secret access key in two lines with appropriate variable names
    """
    def __init__(self, min_confidence=0.85):
        self.client = boto3.client('rekognition')
        self.min_confidence = min_confidence * 100
        self.output_names = None

        self._stack_batch = False

        # a bunch of hacks to make this look like a torch module
        self._modules, self._parameters, self._buffers = {}, {}, {}
        self._backward_hooks, self._forward_hooks, self._forward_pre_hooks = None, None, None

    def forward(self, x):
        """Expects inputs to be either bytes or images."""
        if type(x) == list: assert type(x[0]) == Image.Image
        else: assert type(x) == Image.Image

        # format into dict & convert to bytes if image
        if type(x) == list:
            shapes = [xp.size for xp in x]
            x = [{'Bytes': xp if type(xp) == bytes else convert_img_to_bytes(xp)} for xp in x]
        else:
            shapes = [x.size]
            x = [{'Bytes': x if type(x) == bytes else convert_img_to_bytes(x)}]

        # submit parallel requests
        _client_wrapper = lambda d: self.client.detect_labels(
            Image=d, MinConfidence=self.min_confidence
        )
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_client_wrapper, xp) for xp in x]
        out = [f.result() for f in futures]

        return [self.result_to_detection(out[i], shapes[i]) for i in range(len(out))]

    def result_to_detection(self, res, shape):
        """Converts Amazon Results dict object to torchvision detection dict object"""
        assert self.output_names is not None
        res = res['Labels']
        labels, scores, boxes = [], [], []
        for r in res:
            if 'Instances' in r: 
                instances = [self.instance_to_xyxy(i, shape) for i in r['Instances']]
                num_instances = len(instances)
                boxes.extend(instances)
            else: 
                boxes.append([])
                num_instances = 1
            labels.extend([self.output_names.index(r['Name'])] * num_instances)
            scores.extend([r['Confidence']/100] * num_instances)
        return {
            'labels': convert(labels, out='tensor').int(),
            'boxes': convert(boxes),
            'scores': convert(scores, out='tensor'),
        }

    def instance_to_xyxy(self, instance_dict, shape):
        """Converts Amazon Instance dict object to a 4-tuple of (xmin, ymin, xmax, ymax) format"""
        w = instance_dict['BoundingBox']['Width']
        h = instance_dict['BoundingBox']['Height']
        left = instance_dict['BoundingBox']['Left']
        top = instance_dict['BoundingBox']['Top']
        return [left * shape[0], top * shape[1], (left + w) * shape[0], (top + h) * shape[1]]