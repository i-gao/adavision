import numpy as np
from sklearn.svm import LinearSVC
import adatest

class ConstantModel():
    def __init__(self, label):
        self.label = label
        self.classes_ = np.array([label])
    def predict(self, embeddings):
        if not hasattr(embeddings[0], "__len__"):
            return self.label   
        else:
            return [self.label] * len(embeddings)
    def decision_function(self, embeddings):
        if not hasattr(embeddings[0], "__len__"):
            return 0
        else:
            return np.zeros(len(embeddings))

class TopicLabelingModel:
    """
    A model that predicts if a given test (within a topic) is a "passed" or "failed" test.
    """
    def __init__(self, topic, test_tree):
        self.topic = topic
        self.test_tree = test_tree

        # mask out entries that do not have a pass/fail label
        valid_mask = ~((test_tree["labeler"] == "imputed") | (test_tree["label"] == "topic_marker") | (test_tree["label"] == "off_topic"))
        
        # try and select samples from the current topic
        topic_mask = (test_tree["topic"] == topic) & valid_mask
        
        # if we didn't find enough samples then expand to include subtopics
        if topic_mask.sum() <= 1:
            topic_mask = test_tree["topic"].str.startswith(topic) & valid_mask
        
        # if we still didn't find enough samples then expand to include parent topics
        parts = topic.split("/")
        for i in range(len(parts), 0, -1):
            prefix = "/".join(parts[:i+1])
            if topic_mask.sum() <= 1:
                topic_mask = test_tree["topic"].str.startswith(prefix) & valid_mask
            else:
                break

        # get our features and labels for fitting a model
        strings = list(test_tree["input"][topic_mask]) + list(test_tree["output"][topic_mask])
        labels = list(test_tree["label"][topic_mask])
        unrolled_embeds = adatest.embed(strings)
        embeddings = np.hstack([unrolled_embeds[:len(labels)], unrolled_embeds[len(labels):]])

        # empty test tree
        if len(labels) == 0:
            self.model = ConstantModel("Unknown")

        # constant label topic
        elif len(set(labels)) == 1:
            self.model = ConstantModel(labels[0])
        
        # enough samples to fit a model
        else:
            
            # we are in a highly overparametrized situation, so we use a linear SVC to get "max-margin" based generalization
            # TODO: SML: It seems to me that the SVC seems to do very well as long as there are no "errors" in the data labels. But it will
            # do very poorly if there are errors in the data labels since it will fit them exactly. Perhaps we can help this by
            # ensembling several SVCs together each trained on a different bootstrap sample? This might add the roubustness (against label mismatches)
            # that is lacking with hard-margin SVC fitting (it is also motivated a bit by the connections between SGD and hard-margin SVC fitting, and that
            # in practice SGD works on subsamples of the data so it should be less sensitive to label misspecification).
            self.model = LinearSVC()
            # self.model = LogisticRegression(penalty='l2', random_state=0, C=1.0, solver='lbfgs', max_iter=1000)
            self.model.fit(embeddings, labels)

    def __call__(self, input, output, return_confidence=False):
        """Returns a prediction (string) and confidence score (positive float)"""
        input_is_singleton = type(input) != list
        if type(input) != list: input = [input]
        if type(output) != list: output = [output]
        
        embeddings = [np.hstack(tup) for tup in zip(adatest.embed(input), adatest.embed(output))]
        confidences = self.model.decision_function(embeddings)
        labels = self.model.classes_[(confidences > 0).astype(int)]

        if input_is_singleton: 
            confidences = confidences[0]
            labels = labels[0]
        
        if return_confidence: return labels, np.abs(confidences)
        else: return labels

class TopicMembershipModel:
    """ A model that predicts if a given test fits in a given topic.

    Note that this model only depends on the inputs not the output values for a test.
    """
    def __init__(self, topic, test_tree):
        self.topic = topic
        self.test_tree = test_tree

        # mask out entries that do not have a topic membership label
        valid_mask = ~((test_tree["labeler"] == "imputed") | (test_tree["label"] == "topic_marker"))
        
        # try and select samples from the current topic
        topic_mask = (test_tree["topic"] == topic) & valid_mask
        
        # if we didn't find enough samples then expand to include subtopics
        if topic_mask.sum() <= 1:
            topic_mask = test_tree["topic"].str.startswith(topic) & valid_mask
        
        # if we still didn't find enough samples then expand to include parent topics
        parts = topic.split("/")
        for i in range(len(parts), 0, -1):
            prefix = "/".join(parts[:i+1])
            if topic_mask.sum() <= 1:
                topic_mask = test_tree["topic"].str.startswith(prefix) & valid_mask
            else:
                break

        # get our features and labels for fitting a model
        strings = list(test_tree["input"][topic_mask])
        labels = [l if l == "off_topic" else "on_topic" for l in test_tree["label"][topic_mask]]
        embeddings = np.array(adatest.embed(strings))

        # empty test tree
        if len(labels) == 0:
            self.model = ConstantModel("Unknown")

        # constant label topic
        elif len(set(labels)) == 1:
            self.model = ConstantModel(labels[0])
        
        # enough samples to fit a model
        else:
            
            # we are in a highly overparametrized situation, so we use a linear SVC to get "max-margin" based generalization
            self.model = LinearSVC()
            self.model.fit(embeddings, labels)

    def __call__(self, input, return_confidence=False):
        """Returns a prediction (string) and confidence score (float)"""
        input_is_singleton = type(input) != list
        if type(input) != list: input = [input]

        embeddings = adatest.embed(input)
        confidences = self.model.decision_function(embeddings)
        labels = self.model.classes_[(confidences > 0).astype(int)]

        if input_is_singleton: 
            confidences = confidences[0]
            labels = labels[0]
        
        if return_confidence: return labels, np.abs(confidences)
        else: return labels
