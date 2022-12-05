import numpy as np
import torch
import logging
import shap
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_tensor, to_pil_image
import adatest

log = logging.getLogger(__name__)

class Scorer():
    def __new__(cls, model, *args, **kwargs):
        """ If we are wrapping an object that is already a Scorer, we just return it.
        """
        if shap.utils.safe_isinstance(model, "adatest.Scorer"):
            return model
        else:
            return super().__new__(cls)
    
    def __init__(self, model):
        """ Auto detect the model type and subclass to the right scorer object.
        """

        # ensure we have a model of type Model
        if isinstance(getattr(self, "model", None), Model) or shap.utils.safe_isinstance(getattr(self, "model", None), "shap.models.Model"):
            pass
        elif isinstance(model, Model) or shap.utils.safe_isinstance(model, "shap.models.Model"):
            self.model = model
        else:
            self.model = Model(model)

        # If we are in the base class we need to pick the right specialized subclass to become
        if self.__class__ is Scorer:

            # finish early if we are wrapping an object that is already a Scorer (__new__ will have already done the work)
            if shap.utils.safe_isinstance(model, "adatest.Scorer"):
                return
            
            # see if we are scoring a generator or a classifier
            out = self.model(["string 1", "string 2"])
            if isinstance(out[0], str):
                self.__class__ = GeneratorScorer
                GeneratorScorer.__init__(self, model)
            else:
                self.__class__ = ClassifierScorer
                ClassifierScorer.__init__(self, model)
            
class ClassifierScorer(Scorer):
    """Wraps a classification model and defines a callable scorer that returns a score value for any input.
    Scores are saved in a "model score" column by default. The score is simply the top-1 model softmax confidence.
    This score is displayed in the interface as a colored bar to show model confidence.

    Intended for both NLP and Vision.
    """

    def __init__(self, model, top_probs=20, top_k=1, output_names=None):
        """ 
        Parameters:
        -----------
        model : callable
            A model that is callable with a single argument (which is a list of strings) and returns a matrix of outputs,
            where each row is a size-|Y| probability vector.
            Vision models should take in a list of URLs to images, while text models will receive the list of inputs directly.

        top_probs : int
            The number of top output probabilities to consider when scoring tests. This is used to reduce the number of
            input/output pairs that are passed to the local topic labeling model (and so save compute).

        top_k : int
            The number of top logits to show the user. The user is meant to mark tests as passed if the correct label is 
            in any of the top K predictions. This makes the task easier for models when examples are noisily labeled, e.g.,
            when images contain multiple objects.

        output_names : list of strings
            A list of strings that correspond to the outputs of the model. If None, model.output_names is used.
        """
        super().__init__(model)

        # extract output names from the model if they are not provided directly
        if output_names is None and getattr(self, "output_names", None) is None:
            self.output_names = self.model.output_names
        elif output_names is not None:
            self.output_names = output_names
        elif not hasattr(self, "output_names"):
            self.output_names = None
        
        if not callable(self.output_names):
            self._output_name_to_index = {v: i for i, v in enumerate(self.output_names)}
        self.top_probs = top_probs
        self.top_k = top_k

    def __call__(self, tests, eval_ids):
        """ Compute the scores (and model outputs) for the tests matching the given ids.
        
        Parameters
        ----------
        tests : TestTree
            A test tree for scoring. Note this should be the full test tree since it defines the local topic label
            models used for scoring.

        eval_ids : list of strings
            The ids of the tests to score.
        """        
        # note: pandas does not preserve order when using tests.loc[eval_ids], so we build up a list of inputs ourselves
        eval_inputs = [tests.loc[id, "input"] for id in eval_ids]

        # run the model
        try:
            model_out = self.model([x.replace("__IMAGE=", "") for x in eval_inputs]) # assuming: this outputs softmax probs
        except Exception as e:
            log.error(e)
            return

        top_k_idxs = torch.topk(torch.Tensor(model_out), self.top_k, dim=1).indices
        outputs, scores = [], []
        for i in range(len(model_out)):
            # compute the output strings and scores for each output
            outputs.append(', '.join([self.model.output_names[y] for y in top_k_idxs[i]]))
            scores.append(np.max(model_out[i][top_k_idxs[i]]))

        return outputs,scores,None,None

class DetectionScorer(Scorer):
    """ Wraps an object detection model and defines a callable scorer that returns a score value for any input/output pair.

    Intended for Vision only.
    """

    def __init__(self, model, output_names=None, filter_outputs=None):
        """        
        Parameters:
        -----------
        model : callable
            A model that is callable with a single argument (which is a list of URL strings) and returns three lists: bboxes, preds, confs.

        output_names : list of strings
            A list of strings that correspond to the outputs of the model. If None, model.output_names is used.

        filter_outputs : fn
            If not None, remove detected boxes if they don't pass the filter_outputs function.
            Input: a single label name
            Output: true/false
        """
        super().__init__(model)
        self.filter_outputs = filter_outputs

        # extract output names from the model if they are not provided directly
        if output_names is None and getattr(self, "output_names", None) is None:
            self.output_names = self.model.output_names
        elif output_names is not None:
            self.output_names = output_names
        elif not hasattr(self, "output_names"):
            self.output_names = None
        
        if not callable(self.output_names):
            self._output_name_to_index = {v: i for i, v in enumerate(self.output_names)}

    def __call__(self, tests, eval_ids):
        """ Compute the scores (and model outputs) for the tests matching the given ids.

        Parameters
        ----------
        tests : TestTree
            A test tree for scoring. Note this should be the full test tree since it defines the local topic label
            models used for scoring.

        eval_ids : list of strings
            The ids of the tests to score.
        """
        # note: pandas does not preserve order when using tests.loc[eval_ids], so we build up a list of inputs ourselves
        eval_inputs = [tests.loc[id, "input"] for id in eval_ids]

        # run the model
        try:
            bboxes, preds, confs = self.model([x.replace("__IMAGE=", "") for x in eval_inputs])
        except Exception as e:
            log.error(e)
            return

        out_imgs, outputs, scores = [], [], []
        for i in range(len(eval_inputs)):
            assert len(bboxes[i]) == len(preds[i]) and len(preds[i]) == len(confs[i])

            # convert from integers to strings
            preds[i] = np.array(list(sorted([self.output_names[y] for y in preds[i]])))
            bboxes[i] = np.array(bboxes[i])
            confs[i] = np.array(confs[i])
            
            # filter out any predictions
            if self.filter_outputs is not None:
                mask = np.array([self.filter_outputs(s) for s in preds[i]])
            else:
                mask = np.ones(len(preds[i]), dtype=bool)

            # handle empty case separately
            if len(preds[i]) == 0 or len(preds[i][mask]) == 0:
                outputs.append("")
                scores.append(1.0)
                out_imgs.append(None)
                continue
            
            # otherwise, compute output, score, and save a tmp image with bboxes drawn in
            outputs.append(
                ', '.join(np.unique(preds[i][mask]))
            )
            scores.append(
                adatest.utils.safe_mean(confs[i][mask])
            )
            bboxes[i] = bboxes[i][mask]
            boxes = [bboxes[i][j] for j in range(len(bboxes[i])) if len(bboxes[i][j]) > 0]
            img = adatest.utils.get_image(eval_inputs[i])[0]
            out_imgs.append(to_pil_image(draw_bounding_boxes(
                    (to_tensor(img) * 255).type(torch.uint8), 
                    boxes=torch.Tensor(np.array(boxes)), 
                    colors="cyan", width=np.ceil(img.width / 150).astype(int),
            ).detach()))

        return outputs,scores,out_imgs,self._bboxes_to_str(bboxes)
 
    def _bboxes_to_str(self, outputs):
        return [str([b.tolist() for b in bboxes]) for bboxes in outputs]

class CaptioningScorer(Scorer):
    """ Wraps an image captioning model and defines a callable scorer that returns a score value for any input/output pair.
    Intended for Vision only.
    """

    def __init__(self, model, top_k=1, filter_outputs=None):
        """ 
        Parameters:
        -----------
        model : callable
            A model that is callable with a single argument (which is a list of URL strings) and returns two lists:
            captions and confidences.

        top_k : int
            The maximum number of generated captions to return for each image, where more confident captions will be selected
            before less confident captions. If the model produces fewer captions than top_k, we'll just return all generated captions.
        """
        super().__init__(model)
        self.filter_outputs = filter_outputs
        self.top_k = top_k

    def __call__(self, tests, eval_ids):
        """ Compute the scores (and model outputs) for the tests matching the given ids.

        Parameters
        ----------
        tests : TestTree
            A test tree for scoring. Note this should be the full test tree since it defines the local topic label
            models used for scoring.

        eval_ids : list of strings
            The ids of the tests to score.
        """
        # note: pandas does not preserve order when using tests.loc[eval_ids], so we build up a list of inputs ourselves
        eval_inputs = [tests.loc[id, "input"] for id in eval_ids]

        # run the model
        try:
            captions, confs = self.model([x.replace("__IMAGE=", "") for x in eval_inputs])
        except Exception as e:
            log.error(e)
            return

        outputs, scores = [], []
        for i in range(len(eval_inputs)):
            assert len(captions[i]) == len(confs[i])

            # convert from lists to arrays
            captions[i] = np.array(captions[i])
            confs[i] = np.array(confs[i])

            # handle empty case separately
            if len(captions[i]) == 0:
                outputs.append("[no caption]")
                scores.append(1.0) # this is a bad failure
                continue

            # filter out any predictions
            if self.filter_outputs is not None:
                mask = np.array([self.filter_outputs(s) for s in captions[i]])
            else:
                mask = np.ones(len(captions[i]), dtype=bool)
            
            # otherwise, compute output & score
            top_k_idxs = torch.topk(torch.Tensor(confs[i][mask]), min(self.top_k, len(confs[i][mask]))).indices
            caption = captions[i][mask][top_k_idxs]
            outputs.append('\n'.join(
                caption if type(caption) == list else [caption]
            ))
            scores.append(np.max(
                confs[i][top_k_idxs]
            ))

        return outputs,scores,None,None

#############


class Model():
    """ Wrap all models to have a consistent interface for scoring."""

    def __new__(cls, model, *args, **kwargs):
        """ If we are wrapping a model that is already a Model, we just return it.
        """
        if isinstance(model, Model) or shap.utils.safe_isinstance(model, "shap.models.Model"):
            return model
        else:
            return super().__new__(cls)
    
    def __init__(self, model, output_names=None, **kwargs):
        """ Build a new model by wrapping the given model object.

        Parameters
        ----------
        model : object
            The model to wrap. This can be a plain python function that accepts a list of strings and returns either
            a vector of probabilities or another string. It can also be a transformers pipeline object (we try to wrap
            common model types transparently).

        output_names : list of str, optional
            The names of the outputs of the model. If not given, we try to infer them from the model.
        """

        # finish early if we are wrapping an object that is already a Model
        if isinstance(model, Model) or shap.utils.safe_isinstance(model, "shap.models.Model"):
            if output_names is not None:
                self.output_names = output_names
            assert len(kwargs) == 0
            return

        # get outputs names from the model if it has them and we don't
        if output_names is None and hasattr(model, "output_names"):
            output_names = model.output_names

        # If we are in the base class we check to see if we should rebuild the model as a specialized subclass
        if self.__class__ is Model:        
            self.inner_model = model
            self.output_names = output_names

    def __call__(self, *args, **kwargs):
        return self.inner_model(*args, **kwargs)