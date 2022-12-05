import torch
import mmcv
from PIL import Image
from mmdet.apis import inference_detector
from .utils import convert_img_to_bytes, convert

class MMDetector(torch.nn.Module):
    """Wraps an mmdet model s.t. the forward function plays well with image inputs"""
    def __init__(self, interior_model):
        super().__init__()
        self.interior_model = interior_model
        self.CLASSES = interior_model.CLASSES
        self._stack_batch = False

    def forward(self, x):
        """Expects inputs to be either bytes or images."""
        if type(x) == list: assert type(x[0]) == bytes or type(x[0]) == Image.Image
        else: assert type(x) == bytes or type(x) == Image.Image

        # convert to bytes & then an mmcv array
        if type(x) == list:
            x = [mmcv.imfrombytes(xp) if type(xp) == bytes else mmcv.imfrombytes(convert_img_to_bytes(xp)) for xp in x]
        else:
            x = mmcv.imfrombytes(x) if type(x) == bytes else mmcv.imfrombytes(convert_img_to_bytes(x))

        out = inference_detector(self.interior_model, x) # returns a list of 80 items, one for each class.
        return [self.result_to_detection(r) for r in out]

    def result_to_detection(self, res):
        """Converts an mmdet list of tensors into torchvision detection dict"""
        indices = [i for i in range(len(res)) if len(res[i]) > 0]
        
        labels, boxes, scores = [], [], []
        for i in indices:
            labels.extend([i] * len(res[i]))
            boxes.extend(res[i][:, :-1].tolist()) 
            scores.extend(res[i][:, -1].tolist())

        return {
            'labels': convert(labels, out='tensor').int(),
            'boxes': convert(boxes),
            'scores': convert(scores, out='tensor'),
        }