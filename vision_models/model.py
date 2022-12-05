import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import adatest

from PIL import Image

from .utils import convert, download_img, is_url, sanitize_strings
from .utils import Stopwatch as sw

class Model(torch.nn.Module):
    """
    Wrapper for a vision model
    Optimized for inference speed
    """
    def __init__(self, 
                 model, 
                 transform_list=[], 
                 batch_size=None,
                 output_names=None, 
                 use_amp=False,
                 use_jit=True,
                 download_fn=None,
    ):
        super().__init__()
        self.transform = transforms.Compose(transform_list) if len(transform_list) else None
        self.output_names = sanitize_strings(output_names) if output_names is not None else None
        self.batch_size = batch_size
        self.use_amp = use_amp
        if self.use_amp: 
            print("Warning: mixed precision may result in accuracy loss.")

        # initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.device = self.device
        model = model.to(self.device).eval()
        torch.set_grad_enabled(False)
        self.model = model

        # set behavior for how to deal with URL inputs
        self._download_urls = getattr(self.model, "input_type", "PIL") == "PIL"
        self.download_fn = adatest.utils.get_image # has caching enabled

        # set behavior for whether to pass into the model a list or matrix
        self._stack_batch = getattr(self.model, "_stack_batch", True)

    def forward(self, x):
        """
        Forward pass an image or list of images, where images are either in URL or PIL formats.
        """ 
        # if dealing with urls, and the inner model expects images, download and convert to PIL
        if self._download_urls and (is_url(x) or (type(x) == list and is_url(x[0]))):
            x = self.download_fn(x)
        
        # apply transforms
        if self.transform is not None: 
            if type(x) == list:
                x = [self.transform(xp).to(self.device) for xp in x]
                if self._stack_batch: x = torch.stack(x, dim=0)
            else:
                x = self.transform(x).to(self.device)
                x = x.view(1, *x.shape)

        try: n = len(x)
        except: n = 1
        
        with sw() as FT:
            # forward pass in batches
            ret = []
            batch_size = self.batch_size if self.batch_size else n
            n_batches = (n // batch_size) + int(n % batch_size > 0)

            for i in range(n_batches):
                with torch.autocast(self.device.type, enabled=self.use_amp):
                    out = self.model.forward(x[i*batch_size:(i+1)*batch_size])

                # not sure why, but sometimes FasterRCNN returns a ([{}]) instead of [{}]
                if type(out) == tuple: out = [o for o in out]
                
                if type(out) == list:
                    ret.extend(convert(out, "tensor"))
                else:
                    ret.append(convert(out, "tensor"))

            # collate results
            if torch.is_tensor(ret[0]): 
                ret = torch.vstack(ret)
        
        print(f"Passed images through model in {FT.time}s.")
        return ret

class ClassificationModel(Model):
    """
    Wrapper for a vision classification model
    s.t. the model takes in URLs or PIL images and outputs
    predictions and softmax confidences.
    """
    def __init__(self, model, **model_kwargs):
        super().__init__(
            model=model,
            **model_kwargs
        )

    def prediction(self, x, k=1, return_classname=True):
        """
        Get top-k model predictions and confidences for x
        Let n be the size of the batch.
        Returns:
            - len-n list of dicts [{}], where each dict contains two keys
                - pred: either a k x 1 array of classname strings
                        or a k x 1 array of integers corresponding to classes
                - conf: k x 1 array of floats with softmax value for corresponding class
        """
        out = self.forward(x)
        conf, pred = torch.topk(F.softmax(out, dim=1), k=k, dim=1)

        # output_names
        if return_classname:
            assert self.output_names is not None, "No class names stored in this Model"
            names = []
            for x in pred.tolist():
                names.append([self.output_names[y] for y in x])
            pred = names

        # collate
        ret = [{'pred': convert(pred[i]), 'conf': convert(conf[i])} for i in range(len(out))]
        return ret

    def confidences(self, x):
        """
        Get model class confidences for x. Let K be the total number of classes, i.e. len(m.output_names).
        This function is a simplification of m.prediction(). It is convenient for connecting models to
        the Adatest ClassificationScorer API.
        Returns:
            - n x K array of confidences
        """
        out = self.forward(x)
        return convert(F.softmax(out, dim=1))

    def __call__(self, x):
        """Change default call function to play well with Adatest."""
        return self.confidences(x)

class DetectionModel(Model):
    """
    Wrapper for a vision classification model
    s.t. the model takes in URLs or PIL images and outputs
    bounding boxes, label predictions, and confidences.
    """
    def __init__(self, model, **model_kwargs):
        # most detection models do not want stacked batches
        if not hasattr(model, "_stack_batch"): model._stack_batch = False
        super().__init__(
            model=model,
            **model_kwargs,
        )

    def prediction(self, x, return_classname=True):
        """
        Get model bounding box predictions, label predictions, and confidences for x
        Let n be the size of the batch, k_i be the number of detections in example i
        Returns:
            - len-n list of dicts [{}], where each dict contains three keys
                - pred: either a k_i x 1 array of classname strings
                        or a k_i x 1 array of integers corresponding to classes
                - conf: k_i x 1 array of floats with softmax value for corresponding class
                - bbox: k_i x 4 array of bbox coordinates in (xmin, ymin, xmax, ymax) format
        """
        out = self.forward(x)

        # output_names
        if return_classname:
            assert self.output_names is not None, "No class names stored in this Model"
            for i in range(len(out)):
                out[i]['labels'] = [self.output_names[y] for y in out[i]['labels']]

        # collate
        ret = [{'pred': convert(r['labels']), 'bbox': convert(r['boxes']), 'conf': convert(r['scores'])} for r in out]
        return ret

    def collated_predictions(self, x):
        """
        Get model class confidences for x. Let n be len(x). Let K be the total number of classes, i.e. len(m.output_names).
        Unlike m.confidences(), this function returns bboxes and sums, rather than maxes over, confidences
        of multiple predictions of the same class.
        This function is a simplification of m.prediction(). It is convenient for connecting models to
        the Adatest DetectionScorer API.
        Returns:
            - boxes: len-n list of arrays of bboxes
            - preds: len-n list of arrays of strings corresponding to output names
            - conf: len-n list of arrays of confidences in range [0,1]
        """
        assert self.output_names is not None, "output_names are required for this method in order to know the number of classes"
        out = self.prediction(x, return_classname=False)
        
        boxes, preds, conf = [], [], []
        for i, o in enumerate(out):
            boxes.append(o['bbox'])
            preds.append(o['pred'])
            conf.append(o['conf'])
        return boxes, preds, conf

    def __call__(self, x):
        """Change default call function to play well with Adatest."""
        return self.collated_predictions(x)

class CaptioningModel(Model):
    """
    Wrapper for a vision captioning model
    s.t. the model takes in URLs or PIL images and outputs captions and confidences.
    """
    def __init__(self, model, **model_kwargs):
        super().__init__(
            model=model,
            **model_kwargs,
        )

    def prediction(self, x, return_classname=True):
        """
        Get model bounding box predictions, label predictions, and confidences for x
        Let n be the size of the batch, k_i be the number of captions generated for example i
        Returns:
            - len-n list of dicts [{}], where each dict contains keys
                - pred: either a k_i x 1 array of caption strings
                - conf: k_i x 1 array of floats with confidence value for each caption
        """
        out = self.forward(x)

        # collate
        ret = [{'pred': convert(r['captions']), 'conf': convert(r['scores'])} for r in out]
        return ret

    def confidences(self, x, return_classname=True):
        """
        Get model bounding box predictions, label predictions, and confidences for x
        Let n be the size of the batch, k_i be the number of captions generated for example i
        Returns:
            - captions: len-n list of arrays of captions
            - conf: len-n list of arrays of confidences in range [0,1]
        """
        out = self.forward(x)
        captions, conf = [], []
        for i, o in enumerate(out):
            captions.append(o['captions'])
            conf.append(o['scores'])
        return convert(captions), convert(conf)
    

    def __call__(self, x):
        """Change default call function to play well with Adatest."""
        return self.confidences(x)