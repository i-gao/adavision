import numpy as np
import logging
import re
import urllib.parse
import adatest
from .embedders import cos_sim
from .utils import is_subtopic # note: cannot use adatest.utils here b/c of circular imports
log = logging.getLogger(__name__)


class PromptBuilder():
    """ A class to build prompts for the model.
    """
    
    def __init__(self, prompt_size=3, slot_randomization=0.01, score_randomization=0.05, threshold=0.5, prompt_diversity=True,
                 subtopic_diversity=False):
        """ Initialize the prompt builder.
        
        Parameters
        ----------
        prompt_size : int
            The number of rows to select as prompts.
        
        slot_randomization : float
            The proportion of slots to make fully random (within the current topic). p in Binomial(n,p)

        score_randomization : float
            The standard deviation of an additive Gaussian randomization factor for the scores.

        prompt_diversity : bool
            Whether to include a diversity term when selecting tests for the prompt. This diversity term is based
            on the embeddings of each test.

        subtopic_diversity : bool
            If true, we will try and pick tests from a diverse set of subtopics of the current topic (if we are
            using subtopic tests and not direct child tests).
        """

        self.prompt_size = prompt_size
        self.slot_randomization = slot_randomization
        self.score_randomization = score_randomization
        self.prompt_diversity = prompt_diversity
        self.subtopic_diversity = subtopic_diversity
        self.threshold = threshold
    
    def __call__(self, test_tree, topic, score_column, repetitions=1, filter="", suggest_topics=False, working_set_size=100):
        """ This selects up to self.prompt_size rows in the test tree which will be used to prompt the generator.
        The overall strategy is to score every row in the test tree, and then select the highest scoring tests/subtopics
        from within the current topic.

        Score are based off of the model confidence and the test label, but to promote diversity, we add some gaussian noise to scores, enforce
        that prompt choices are sufficiently diverse among each other (in terms of cosine embedding similarity), and choose some of the prompts randomly
        rather than greedily.

        Under the hood, what we do is compute a score vector in range [0,1], and then elementwise multiply the score vector by a bunch of other 
        vectors, often binary, to mask out / prompt / demote certain rows.

        Parameters
        ----------
        test_tree : adatest.TestTree
            The test tree to generate prompts from.
        
        topic : str
            The topic to build a prompt for.

        score_column : str
            The column to use for scoring the tests.

        repetitions : int
            The return value is a list of len repetitions, each containing self.prompt_size lists.
            In practice, repetitions corresponds to the Test Tree Browser's number of suggestion_threads.
            In this project, we just use a single suggestion_thread, so repetitions is always = 1.
            See the comment in _test_tree.py:adapt() for detailed notes.
        
        filter : str
            A filter to apply to the test set before selecting tests to build the prompt.

        suggest_topics : bool
            If true, we will create a prompt filled with topic names instead of a list of tests.

        working_set_size : int
            How many top tests to consider when doing the full iterative scoring process. Larger values may take longer.
            Note that this has no effect as long as we never go more than working_set_size tests deep during the prompt
            item selection process.

        Returns:
            [[(,)]], where the outer list is of len repetitions; the inner lists are of len up to self.prompt_size, 
            and each prompt is a tuple of (id, topic, input) when not suggest_topics or (id, parent topic, topic name)
            when suggest_topics
        
        """

        ids = np.array(test_tree.index)

        # return early for an empty test tree
        if len(ids) == 0:
            return [[] for _ in range(repetitions)]

        # hide rows that don't match the filter
        hidden_scaling = np.ones(len(ids))
        if filter != "":
            filter_compiled = re.compile(filter)
            for i,k in enumerate(ids):
                test = test_tree.loc[k]
                if hasattr(test, "input") and not test.input.startswith("__IMAGE=") and filter_compiled.search(test.input) is not None:
                    continue
                if hasattr(test, "output") and filter_compiled.search(test.output) is not None:
                    continue
                hidden_scaling[i] = 0.0

        # when suggesting tests, only build prompts using tests, not topics
        # and vice versa when suggesting topics
        # never use suggestions
        if suggest_topics:
            type_scaling = np.array(test_tree["label"] == "topic_marker", dtype=int)
        else:
            type_scaling = np.array(test_tree["label"] != "topic_marker", dtype=int)
        type_scaling *= np.array(["__suggestions__" not in t for t in test_tree["topic"]], dtype=int)

        # we want to select prompts from tests/subtopics within the current topic
        # to do so, we'll zero out the scores of all rows not in the current topic
        # we'll also promote direct children over subtopic descendants
        topic_scaling = np.zeros(len(ids))
        children_ids = test_tree.get_children_in_topic(topic, direct_children_only=False, include_suggestions=False, include_topics=True).index
        topic_scaling[np.isin(ids, children_ids)] = 1
        direct_children_ids = test_tree.get_children_in_topic(topic, direct_children_only=True, include_suggestions=False, include_topics=True).index
        topic_scaling[np.isin(ids, direct_children_ids)] = 100
        topic_scaling /= np.max(topic_scaling) # values between [0, 1]
        
        # return early if we have nothing to build a prompt with
        if np.sum(topic_scaling * hidden_scaling * type_scaling) == 0:
            return [[] for _ in range(repetitions)]

        # compute positive scalar scores for all rows in the test tree
        # this is based on the model confidence and the label
        scores = test_tree.get_scores(score_column)
        
        # hard avoid anything with a flat 0 or nan score before noising
        hard_avoidance = np.array(scores == 0, dtype=int)

        # filter down to just top rows we will use during the iterative scoring process
        rank_vals = scores * topic_scaling * hidden_scaling * type_scaling
        top_inds = np.argsort(-rank_vals)[:working_set_size]
        ids = ids[top_inds]
        topic_scaling = topic_scaling[top_inds]
        hidden_scaling = hidden_scaling[top_inds]
        type_scaling = type_scaling[top_inds]
        scores = scores[top_inds]
        hard_avoidance = hard_avoidance[top_inds]

        # build a list of randomized prompts
        prompts = []
        for _ in range(repetitions):

            diversity = np.ones(len(ids))

            # store tmp versions of things we update during the iteration
            # these are updated via score noising, slot randomization, etc.
            scores_curr = scores.copy()
            topic_scaling_curr = topic_scaling.copy()
            hard_avoidance_curr = hard_avoidance.copy()

            # score noising
            scores_curr = np.maximum(0, scores_curr + np.random.normal(loc=0, scale=self.score_randomization, size=len(ids)))

            # compute how many greedy and how many random positions we will have
            num_random = max(0, min(np.random.binomial(self.prompt_size, self.slot_randomization), len(ids) - self.prompt_size))
            num_greedy = max(0, min(self.prompt_size - num_random, len(ids) - num_random))

            # sim_avoidance is a vector we use to avoid prompts similar to already selected prompts
            if self.prompt_diversity:
                sim_avoidance = np.zeros(len(ids))
                if suggest_topics:
                    embeddings_arr = np.vstack(adatest.embed(
                        [urllib.parse.unquote(test_tree.loc[id, "topic"].split("/")[-1]) for id in ids]
                    ))
                else:
                    embeddings_arr = np.hstack([
                        np.vstack(adatest.embed([test_tree.loc[id, "input"] for id in ids])),
                        np.vstack(adatest.embed([test_tree.loc[id, "output"] for id in ids]))
                    ])
                similarities = cos_sim(embeddings_arr, embeddings_arr)
            
            # iteratively select prompt items
            num_below_threshold = 0
            prompt_ids = []
            outside_topics_used = np.ones(len(ids))
            while len(prompt_ids) < num_greedy + num_random:

                # once we get to the random part of the process we scramble the scores
                if len(prompt_ids) == num_greedy:
                    scores_curr = 1 + np.random.rand(len(ids))*0.1

                # find the next bext index
                if self.prompt_diversity:
                    diversity = 1 - (similarities * sim_avoidance).max(1)
                rank_vals = scores_curr * topic_scaling_curr * diversity * (1 - hard_avoidance_curr) * hidden_scaling * type_scaling * outside_topics_used

                if np.isnan(np.min(rank_vals)) or (np.nanmax(rank_vals) <= 0 and len(prompt_ids) > 0): # stop if we have run out of the current subtree
                    break

                new_ind = np.nanargmax(rank_vals)
                score = scores_curr[new_ind]

                # allow at most num_random prompts to have a scores_curr < self.threshold
                # if too many choices have < self.threshold score, then just reduce the number of prompts returned
                if score < self.threshold:
                    num_below_threshold += 1
                if num_below_threshold > num_random:
                    break

                # make it unlikely we will choose the same outside topic twice
                new_ind_topic = test_tree.loc[ids[new_ind], "topic"]
                if not is_subtopic(topic, new_ind_topic):
                    outside_topics_used *= 1 - 0.9 * np.array([test_tree.loc[id, "topic"] == new_ind_topic for id in ids])

                # add this item
                prompt_ids.append(ids[new_ind])
                avoidance_level = 1

                # avoid this IO pair as we select the next pairs
                # also keep track in sim_avoidance s.t. it affects our diversity calculations above
                hard_avoidance_curr[new_ind] = avoidance_level
                if self.prompt_diversity:
                    sim_avoidance[new_ind] = avoidance_level

                # lower the weight of the subtopic we just picked from
                if self.subtopic_diversity:
                    if topic != new_ind_topic and is_subtopic(topic, new_ind_topic):
                        subtopic = topic + "/" + new_ind_topic[(len(topic)+1):].split("/")[0]
                        subtopic_scaling = np.array([0.001 if is_subtopic(subtopic, test_tree.loc[k, "topic"]) else 1 for k in ids])
                        topic_scaling_curr *= subtopic_scaling

            # create the prompt as a list of tuples
            prompt = []
            for k in reversed(prompt_ids):
                row = test_tree.loc[k]
                if suggest_topics:
                    if row["topic"] == "":
                        continue # we can't use the root to help suggest topic names
                    parents,child = row["topic"].rsplit("/", 1)
                    prompt.append((k, parents, urllib.parse.unquote(child)))
                else:
                    prompt.append((k, row["topic"], row["input"]))
            prompts.append(prompt)

        return prompts