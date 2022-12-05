import torchvision
import json
import os
import numpy as np
import pandas as pd
import re
from .model import ClassificationModel, DetectionModel, CaptioningModel
from .utils import getattr_recursive
from torchvision import transforms
from PIL import Image
import pathlib
file_path = str(pathlib.Path(__file__).parent.resolve())

def list_models(filt_exp=None):
    """
    List all models that are loadable.
    """
    # search through globals for all _list_X_models function
    # each load_X_model function should have a corresponding _list_X_models function to appear in this function's output
    list_fns = [k for k in globals() if re.match("_list_(.+)_models", k)]

    # create a dict mapping from all model names to the appropriate loading function and the minimum loading function kwargs to select this model
    # format: {model_name: (load_fn, task, load_fn_kwargs)}
    out = {}
    for l in list_fns:
        out.update(globals()[l]())

    if filt_exp is not None:
        out = dict(filter(lambda i: re.match(f".*{filt_exp}.*", i[0]), out.items()))
    
    return out

#############

def load_torchvision_model(model_name, torchvision_weights=None, task='classification', torchvision_kwargs={}, **model_kwargs):
    """
    Load a pretrained torchvision model by name
    Args:
        - model_name {str} name of model
        - torchvision_weights {str} name of a torchvision WeightsEnum object, which references pretrained weights,
            the pretraining output_names, expected input size, etc., as trained by torchvision. Available WeightsEnums
            are listed at https://pytorch.org/vision/stable/models.html. If this variable is set,
            it overrides any 'weight' key in torchvision_kwargs.
            If this variable is not set, the 'DEFAULT' pretrained weights for the model will be used.
        - torchvision_kwargs {dict} any kwargs to pass in when initializing the model
        - model_kwargs: remaining kwargs to pass into the models.Model initializer, including:
            - transform_list {list} augmentations to apply to examples passed into the model
            - output_names {list}
            - batch_size {int}
    """
    if task == 'classification':
        import_src = torchvision.models
    elif task == 'detection':
        import_src = torchvision.models.detection

    # parse torchvision_weights
    if torchvision_weights is not None: 
        torchvision_weights = getattr_recursive(import_src, torchvision_weights)
        model_kwargs['output_names'] = torchvision_weights.meta['categories']
        min_size = torchvision_weights.meta['min_size']
    else:
        torchvision_weights = torchvision_kwargs.pop('weights', 'DEFAULT')
        model_kwargs['output_names'], min_size = None, None

    # parse / check transform_list
    if torchvision_weights is not None and 'transform_list' not in model_kwargs:
        model_kwargs['transform_list'] = [torchvision_weights.transforms()]
    else:
        if not _check_transform_outputs_tensor(model_kwargs['transform_list']):
            model_kwargs['transform_list'] = [*model_kwargs['transform_list'], *DEFAULT_TRANSFORMS]
        assert _check_transform_meets_min_size(model_kwargs['transform_list'], min_size), f"{model_name} requires images to be transformed to minimum size {min_size}"
        
    # fetch model
    model = getattr_recursive(import_src, model_name)(**torchvision_kwargs, weights=torchvision_weights)

    if task == 'classification':
        constructor = ClassificationModel
        if model_kwargs['output_names'] is None: 
            model_kwargs['output_names'] = list(json.load(open(file_path + "/output_names/imagenet1K.txt")).values())
    elif task == 'detection':
        constructor = DetectionModel
        if model_kwargs['output_names'] is None: 
            model_kwargs['output_names'] = list(json.load(open(file_path + "/output_names/mscoco.txt")).values())
    else:
        raise ValueError(f'Model: {model_name} not recognized.')

    packaged_model = constructor(
        model, **model_kwargs
    )

    return packaged_model

def _list_torchvision_models():
    import torchvision

    def _extract_models(keys):
        weights = [k for k in keys if re.match("(.+)_Weights", k)]
        models = [re.match("(.+)_Weights", k).group(1).lower() for k in weights]
        return list(zip(models, weights))

    classification = {
        m: (
            load_torchvision_model,
            "classification",
            {'model_name': m, 'torchvision_weights': w, 'task': "classification"}
        )
        for m, w in _extract_models(torchvision.models.__dict__.keys())
    }
    detection = {
        m: (
            load_torchvision_model,
            "detection",
            {'model_name': m, 'torchvision_weights': w, 'task': "detection"}
        )
        for m, w in _extract_models(torchvision.models.detection.__dict__.keys())
    }
    return {**classification, **detection}

def load_timm_model(model_name, **model_kwargs):
    """
    Load a pretrained timm model by name
    Args:
        - model_name {str} name of model
        - torchvision_kwargs {dict} any kwargs to pass in when initializing the model
        - model_kwargs: remaining kwargs to pass into the models.Model initializer, including:
            - transform_list {list} augmentations to apply to examples passed into the model
            - output_names {list}
            - batch_size {int}
    """
    import timm

    model = timm.create_model(model_name, pretrained=True)

    # parse / check transform_list
    config = model.pretrained_cfg
    min_size = config['input_size'][1:]
    if 'transform_list' not in model_kwargs:
        model_kwargs['transform_list'] = [transforms.Resize(min_size), *DEFAULT_TRANSFORMS]
    else:
        if not _check_transform_outputs_tensor(model_kwargs['transform_list']):
            model_kwargs['transform_list'] = [*model_kwargs['transform_list'], *DEFAULT_TRANSFORMS]
        assert _check_transform_meets_min_size(model_kwargs['transform_list'], min_size), f"{model_name} requires images to be transformed to minimum size {min_size}"
    
    # parse output_names
    if 'output_names' not in model_kwargs:
        model_kwargs['output_names'] = list(json.load(open(file_path + "/output_names/imagenet1K.txt")).values())
    
    # package and return
    packaged_model = ClassificationModel(
        model, **model_kwargs
    )

    return packaged_model

def _list_timm_models():
    import timm

    return {
        m: (
            load_timm_model,
            "classification",
            {'model_name': m}
        )
        for m in timm.list_models(pretrained=True)
    }

def load_mmdet_model(model_name, min_confidence=0.85, **model_kwargs):
    """
    Load a pretrained mmdetection model by name
    """
    import mim
    from mmdet.apis import init_detector
    from .mmdetector import MMDetector
    
    # download model & config
    if not os.path.exists(f"{file_path}/mmdet/{model_name}.pth") or not os.path.exists(f"{file_path}/mmdet/{model_name}.py"):
        os.makedirs(f"{file_path}/mmdet/", exist_ok=True)
        weight_path = mim.download("mmdet", configs=[model_name], dest_root=f"{file_path}/mmdet")
        os.rename(f"{file_path}/mmdet/{weight_path[0]}", f"{file_path}/mmdet/{model_name}.pth")

    model = MMDetector(init_detector(
        f"{file_path}/mmdet/{model_name}.py",
        f"{file_path}/mmdet/{model_name}.pth",
        cfg_options = {'model.test_cfg.score_thr': min_confidence}
    ))
    model_kwargs['output_names'] = model.CLASSES

    packaged_model = DetectionModel(
        model, **model_kwargs
    )

    return packaged_model

def _list_mmdet_models():
    manual_list = [
        'yolox_x_8x8_300e_coco',
        'vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco',
        'scnet_x101_64x4d_fpn_20e_coco',
        'faster_rcnn_x101_64x4d_fpn_1x_coco',
        'mask_rcnn_x101_32x8d_fpn_mstrain-poly_3x_coco',
        'retinanet_r50_fpn_1x_coco',
        'cascade_mask_rcnn_r50_fpn_mstrain_3x_coco',
        'htc_r50_fpn_1x_coco',
        'deformable_detr_refine_r50_16x2_50e_coco',
        'yolox_s_8x8_300e_coco',
        'queryinst_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco',
    ]
    return {
        m: (
            load_mmdet_model,
            "detection",
            {'model_name': m}
        )
        for m in manual_list
    }

def load_clip_classifier_model(model_name="ViT-L/14", prompt=lambda y: f"an image of a {y}", **model_kwargs):
    """
    Load 0-shot pretrained CLIP model by name
    Args:
        - model_name {str} name of model
        - prompt {function} lambda to map a classstring to another string
        - model_kwargs: remaining kwargs to pass into the models.Model initializer, including:
            - output_names {list}
            - batch_size {int}
    """
    from .clip_classifier import CLIPClassifier

    # parse output_names
    if 'output_names' not in model_kwargs:
        model_kwargs['output_names'] = list(json.load(open(file_path + "/output_names/imagenet1K.txt")).values())

    model = CLIPClassifier(
        model_name=model_name,
        prompt=prompt,
        output_names=model_kwargs['output_names'], 
    )

    packaged_model = ClassificationModel(
        model, transform_list=[model.clip_preprocess], **model_kwargs
    )

    return packaged_model

def _list_clip_classifier_models():
    import clip

    return {
        m: (
            load_clip_classifier_model,
            "classification",
            {'model_name': m}
        )
        for m in clip.available_models()
    }

def load_amazon_rekognition_model(min_confidence=0.85, **model_kwargs):
    """Load Amazon Rekognition"""
    from .rekognition import AmazonRekognition
    
    model = AmazonRekognition(min_confidence=min_confidence)

    # parse output_names
    if 'output_names' not in model_kwargs:
        model_kwargs['output_names'] = sorted(set(pd.read_csv(file_path + "/output_names/amazon_rekognition.csv", header=None)[0]))
 
    model.output_names = model_kwargs['output_names']

    packaged_model = DetectionModel(
        model, **model_kwargs
    )

    return packaged_model

def _list_amazon_rekognition_models():
    return {
        'amazon_rekognition': (
            load_amazon_rekognition_model,
            "detection",
            {}
        )
    }

def load_gcloud_vision_model(task='detection', **model_kwargs):
    """Load Google Cloud Vision API model."""
    from .gcloud_vision import GoogleCloudVisionClassifier, GoogleCloudVisionDetector

    mid_to_classname_dict = pd.read_csv(file_path + "/output_names/gcloud_vision.csv", header=None, index_col=0).to_dict()[1]
    model_kwargs['output_names'] = list(mid_to_classname_dict.values())

    if task == 'classification':
        model = GoogleCloudVisionClassifier(mids=list(mid_to_classname_dict.keys()))
        constructor = ClassificationModel
    elif task == 'detection':
        model = GoogleCloudVisionDetector(mids=list(mid_to_classname_dict.keys()))
        constructor = DetectionModel

    packaged_model = constructor(
        model, **model_kwargs
    )

    return packaged_model

def _list_gcloud_vision_models():
    return {
        'gcloud_vision_classifier': (
            load_gcloud_vision_model,
            "classification",
            {'task': 'classification'}
        ),
        'gcloud_vision_detector': (
            load_gcloud_vision_model,
            "detection",
            {'task': 'detection'}
        ),
    }

def load_azure_cognitive_model(task='captioning', azure_kwargs={}, **model_kwargs):
    """Load Google Cloud Vision API model."""
    from .azure_cognitive import AzureCognitiveCaptioner, AzureCognitiveDetector

    if task == 'captioning':
        model = AzureCognitiveCaptioner(**azure_kwargs)
        constructor = CaptioningModel
    elif task == 'detection':
        model_kwargs['output_names'] = list(mid_to_classname_dict.values())
        model = AzureCognitiveDetector(mids=list(mid_to_classname_dict.keys()))
        constructor = DetectionModel

    packaged_model = constructor(
        model, **model_kwargs
    )

    return packaged_model

def _list_azure_cognitive_models():
    return {
        'azure_cognitive_captioner': (
            load_azure_cognitive_model,
            "captioning",
            {'task': 'captioning'}
        ),
        'azure_cognitive_detector': (
            load_azure_cognitive_model,
            "detection",
            {'task': 'detection'}
        ),
    }

def load_ofa_model(ofa_kwargs={}, **model_kwargs):
    """Load Google Cloud Vision API model."""
    from .ofa import OFACaptioner
    model = OFACaptioner(**ofa_kwargs)

    # use the recommended OFA transform
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    resolution = 480
    model_kwargs['transform_list'] = [
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)
    ]

    packaged_model = CaptioningModel(
        model, **model_kwargs
    )
    return packaged_model

def _list_ofa_models():
    return {
        'ofa': (
            load_ofa_model,
            "captioning",
            {}
        ),
    }
    

###############

_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]
DEFAULT_TRANSFORMS = [
    transforms.ToTensor(),
    transforms.Normalize(
        _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN,
        _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD,
    ),
]

def _check_transform_outputs_tensor(transform_list):
    """check that the transforms outputs a tensor"""
    x = Image.fromarray(np.zeros((2, 2, 3), dtype='uint8'))
    x = transforms.Compose(transform_list)(x)
    return torch.is_tensor(x)

def _check_transform_meets_min_size(transform_list, min_size):
    """check that the transforms outputs a tensor of W, H >= the minimum size"""
    if min_size is None or min_size == (1, 1): 
        return True
    
    if type(min_size) == int: min_size = (min_size, min_size)
    x = Image.fromarray(np.zeros((
        min_size[0] - 1,
        min_size[1] - 1,
        3
    ), dtype='uint8'))
    x = transforms.Compose(transform_list)(x)
    result_size = x.shape # assumes is tensor
    return (result_size[0] >= min_size[0]) and (result_size[1] >= min_size[1])