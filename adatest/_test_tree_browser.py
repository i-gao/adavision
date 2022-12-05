import numpy as np
import pandas as pd
import json
import re
from datetime import datetime
import os

from .comm import JupyterComm
import uuid
import pathlib
import re
import logging
from threading import Timer
from ._scorer import Scorer
import adatest # Need to import like this to prevent circular dependencies

log = logging.getLogger(__name__)

FRONTEND_BOILERPLATE = {"display_parts": [{
    "d_text1a": '"',
    "d_value1": "{}",
    "d_text1b": '"',
    "d_text2a": '"',
    "d_value2": "{}",
    "d_text2b": '"',
    "d_text3a": '',
    "d_value3": "",
    "d_text3b": ''
}]}


class TestTreeBrowser():
    """ Used for browsing and expanding a test tree.
    """
    def __init__(self, test_tree, scorer, generators, user, auto_save,
                 max_suggestions, suggestion_thread_budget, prompt_builder, active_generator,
                 disable_topic_suggestions):
        """ Initialize the TestTreeBrowser.
        TODO: re-introduce a recompute_scores function?
        TODO: document: can have multiple generators, but only one scorer.
        See the __call__ method of TreeBrowser for parameter documentation.
        """
        self.test_tree = test_tree
        self.scorer = scorer
        self.generators = generators
        self.user = user
        self.auto_save = auto_save
        self.max_suggestions = max_suggestions
        self.suggestion_thread_budget = suggestion_thread_budget
        self.prompt_builder = prompt_builder
        self.active_generator = active_generator
        self.filter_text = ""
        self.current_topic = ""
        self.disable_topic_suggestions = disable_topic_suggestions

        self._init_generators()
        self._init_scorer()

        # a unique identifier for this test set instance, used for UI connections
        self._id = uuid.uuid4().hex

        # init temporary states for the server
        self._hidden_topics = {}
        self.comm = None

        # set the initial testing mode (of the topic ""), and set up the supported modes
        if not self.disable_topic_suggestions:
            self.mode = "tests" if self.test_tree.shape[0] > 0 else "topics"
            self.mode_options = lambda topic: ["tests", "topics"]
        else:
            self.mode = "tests"
            self.mode_options = lambda topic: ["tests"]

        # mark static_dir as None, may be overwritten by server
        self._static_dir = None
        self._auto_save() # save the current state of the test tree
        self._suggestions_error = "" # tracks if we failed to generate suggestions

    def _init_generators(self):
        """set up the generators"""
        # convert single generator to the multi-generator dictionary format
        if not isinstance(self.generators, dict):
            self.generators = {'generator': self.generators}
        # merge in default generators into generators
        if adatest.default_generators is not None:
            self.generators = {**adatest.default_generators, **self.generators}

        # get a reference to the active generator
        if self.active_generator == "default":
            if isinstance(self.generators, dict):
                self._active_generator_obj = next(iter(self.generators.items()))[1]
            else:
                self._active_generator_obj = self.generators
        else:
            self._active_generator_obj = self.generators[self.active_generator]
    
    def _init_scorer(self):
        """set up the scorer"""
        ### set up the single scorer (which wraps a model) ###
        if self.scorer is None:
            self.score_columns = []
            return
        
        if not isinstance(self.scorer, dict):
            self.scorer = {"model": self.scorer}
            self.score_columns = ["model score"]
        for k in self.scorer:
            # enforce that the scorer is of the Scorer class
            if not adatest.utils.isinstance_ipython(self.scorer[k], Scorer):
                self.scorer[k] = Scorer(self.scorer[k])

        for c in self.score_columns:
            if c not in self.test_tree.columns:
                self.test_tree[c] = ["__TOEVAL__" for _ in range(self.test_tree.shape[0])]
        
        assert len(self.scorer) == 1


    def _repr_html_(self, prefix="", environment="jupyter", websocket_server=None):
        """ Returns the HTML interface for this browser.

        Parameters
        ----------
        prefix : str
            The URL prefix this test tree browser is being served from.

        environment : str
            The environment this test tree browser is being served from (jupyter or web).
        """

        # spin up a JupyterComm object if we are called directly (which we assume is in a notebook)
        if self.comm is None and environment == "jupyter":
            self.comm = JupyterComm(f'adatest_interface_target_{self._id}', self.interface_event)

        # dump the client javascript to the interface
        file_path = pathlib.Path(__file__).parent.absolute()
        with open(file_path / "resources" / "main.js", encoding="utf-8") as f:
            js_data = f.read()
        interface_html = f"""
<div id="adatest_container_{self._id}" style="width: 100%; all: initial;"></div>
<script type='text/javascript'>
  {js_data};
  AdaTestReactDOM.render(
    AdaTestReact.createElement(AdaTest, {{
      interfaceId: "{self._id}", environment: "{environment}", startingTopic: "{self.current_topic}", prefix: "{prefix}",
      websocket_server: {"undefined" if websocket_server is None else '"'+websocket_server+'"'},\
    }}, null),
    document.getElementById('adatest_container_{self._id}')
  );
</script>
"""
        return interface_html

    def display(self):
        """ Manually display the HTML interface.
        """
        from IPython.display import display, HTML
        display(HTML(self._repr_html_()))

    def _auto_save(self):
        """ Save the current state of the model if we are auto saving.
        """
        if self.auto_save:
            self.test_tree.to_csv()

    def interface_event(self, msg):
        """ Handle interface events from the client.
        Each (if action == _) block checks for a type of event that the client may pass us

        Parameters
        ----------
        msg : dict
            The event messages from the client. Each key in the dictionary is a separate message to either the row
            specified by the key or to whole browser object if the key is 'browser'.
        """

        log.debug(f"interface_event({msg})")

        # loop over each event message
        for k in msg:

            # updating which screen we're on / filtering what's on the screen
            if k == "browser":
                action = msg[k].get("action", None)
                
                # rewdraw the entire interface
                if action == "redraw":
                    # handle edge case where we prohibit tests in root but on first load of root, tests is default
                    if self.mode not in self.mode_options(self.current_topic): 
                        self._change_mode(self.mode_options(self.current_topic)[0])
                    self._refresh_interface()
                
                # generate a new set of suggested tests/topics
                elif action == "generate_suggestions":
                    self._clear_suggestions()
                    self._refresh_interface(loading_suggestions=True)
                    self.test_tree.retrain_topic_labeling_model(self.current_topic)
                    self.test_tree.retrain_topic_membership_model(self.current_topic)
                    self._generate_suggestions(filter=self.filter_text)
                    self._refresh_interface()
                
                # change the current topic
                elif action == "change_topic":
                    new_topic_name = adatest.utils.sanitize_topic_name(msg[k]["topic"])
                    if new_topic_name not in self.test_tree.get_topics().values:
                        # edge case: user has requested a topic that is not a created topic in our test tree, e.g., by directly editing window.location
                        # create this topic
                        print(f"User navigated to a topic not in tree; creating topic {new_topic_name}")
                        self.test_tree.loc[uuid.uuid4().hex] = {
                            "topic": new_topic_name,
                            "label": "topic_marker",
                            "author": self.user,
                            "input": "", 
                            "input_display": "",
                            "output": "",
                            "labeler": self.user,
                            "description": "",
                            "create_time": str(datetime.now()),
                        }
                        self._compute_embeddings_and_scores(self.test_tree)
                        self._auto_save()
                    
                    self.current_topic = new_topic_name
                    self._change_mode(self.mode_options(self.current_topic)[0])
                    self._refresh_interface()
                
                # clear the current set of suggestions
                elif action == "clear_suggestions":
                    self._clear_suggestions()
                    self._refresh_interface()

                # add a new empty subtopic to the current topic
                elif action == "add_new_topic":
                    self.test_tree.loc[uuid.uuid4().hex] = {
                        "topic": self.current_topic + "/" + adatest.utils.sanitize_topic_name("New topic"),
                        "label": "topic_marker",
                        "author": self.user,
                        "input": "", 
                        "input_display": "",
                        "output": "",
                        "labeler": self.user,
                        "description": "",
                        "create_time": str(datetime.now()),
                    }
                    self._compute_embeddings_and_scores(self.test_tree)
                    self._auto_save()
                    self._refresh_interface()

                # change which generator is active
                elif action is None and "active_generator" in msg[k]:
                    self.active_generator = msg[k]["active_generator"]
                    self._active_generator_obj = self.generators[self.active_generator]

                # change between topics and tests
                elif action is None and "mode" in msg[k]:
                    self._change_mode(msg[k]["mode"])
                    self._refresh_interface() # to update active generator

                # edit a topic description
                elif action == 'change_description':
                    id = msg[k]['topic_marker_id']
                    if id not in self.test_tree.index: # TODO: I think this is kind of dangerous...
                        self.test_tree.loc[id, 'topic'] = "" # only the root topic would be missing from the tree
                        self.test_tree.loc[id, 'input'] = ""
                        self.test_tree.loc[id, 'output'] = ""
                        self.test_tree.loc[id, 'label'] = "topic_marker"
                    self.test_tree.loc[id, 'description'] = msg[k]['description']
                    self._auto_save()

                # edit the filter
                elif action == 'change_filter':
                    print("change_filter")
                    self.filter_text = msg[k]['filter_text']
                    self._refresh_interface()


            # updating the label, input, or output of a test {'id': {'label': 'pass', 'labeler': 'user'}}
            elif "topic" not in msg[k]:
                sendback_data = {}
                
                # update the row and recompute scores
                for k2 in msg[k]:
                    self.test_tree.loc[k, k2] = msg[k][k2]
                
                # additional actions
                # if user modified input or output: we need to re-embed the new inputs/outputs & change the model score column
                if "input" in msg[k] or "output" in msg[k]:
                    self.test_tree.loc[k, self.score_columns] = "__TOEVAL__"
                    self._compute_embeddings_and_scores(self.test_tree, user_overwrote_output=("output" in msg[k]))

                # if user modified label (e.g., marking a suggestion as pass/fail)
                elif "label" in msg[k]:
                    self.test_tree.loc[k, "label_confidence"] = 1.0

                    if type(self.test_tree.loc[k, "topic"]) != str:
                        # TODO: figure out when exactly this happens
                        # for now, just assume that this should go in the current topic (which might be dangerous)
                        self.test_tree.loc[k, "topic"] = self.current_topic
                    else:
                        self.test_tree.loc[k, "topic"] = self.test_tree.loc[k, "topic"].replace("/__suggestions__", "") # TODO: this causes failures

                # send just the data that changed back to the frontend
                sendback_data["scores"] = {c: [[k, v] for v in ui_score_parts(self.test_tree.loc[k, c], self.test_tree.loc[k, "label"])] for c in self.score_columns}
                outputs = {c: [[k, json.loads(self.test_tree.loc[k].get(c.replace(" score", "") + " raw outputs", "{}"))]] for c in self.score_columns}
                sendback_data["raw_outputs"] = outputs
                sendback_data["output"] = self.test_tree.loc[k, "output"] # the client may have written [model output]
                sendback_data["label"] = self.test_tree.loc[k, "label"]
                sendback_data["labeler"] = self.test_tree.loc[k, "labeler"]
                sendback_data["topic"] = self.test_tree.loc[k, "topic"]
                sendback_data.update(FRONTEND_BOILERPLATE)
                self.comm.send({k: sendback_data})
                self._auto_save()

            # deleting a test {'id': {'topic': '__DELETE__'}}
            # or moving a test to a different topic
            elif k in self.test_tree.index and "topic" in msg[k] and len(msg[k]) == 1:
                if msg[k]["topic"] == "_DELETE_":
                    self.test_tree.drop(k, inplace=True)
                else:
                    self.test_tree.loc[k, "topic"] = msg[k]["topic"]
                self._compute_embeddings_and_scores(self.test_tree)
                self._refresh_interface()
                self._auto_save()

            # modifying a topic (rename or deletion) {'old topic name': {'topic': 'new topic name'}}
            elif "topic" in msg[k] and len(msg[k]) == 1:
                children = self.test_tree.get_children_in_topic(k, include_self=True, include_suggestions=True) # this is missing subtopics

                # nothing actually happened
                if k == msg[k]["topic"]:
                    return
                
                # deleting topic
                elif msg[k]["topic"] == "_DELETE_":
                    self.test_tree.drop(children.index, inplace=True)
                
                # renaming or moving a topic
                elif k != msg[k]["topic"]:
                    new_name = adatest.utils.sanitize_topic_name(msg[k]["topic"])

                    # prohibit empty names -- keep the current name
                    if new_name == "/": new_name = k

                    # if we are renaming, not moving, also change the author of the topic
                    if k.split("/")[-1] not in msg[k]["topic"]: 
                        my_id = children.loc[(children['topic'] == k) & (children['label'] == 'topic_marker')].index
                        self.test_tree.loc[my_id, "author"] = self.user

                    for id in children.index:
                        self.test_tree.loc[id, "topic"] = self.test_tree.loc[id, "topic"].replace(k, new_name)

                # Recompute any missing embeddings to handle any changes
                self._compute_embeddings_and_scores(self.test_tree)
                self._refresh_interface()
                self._auto_save()

            else:
                log.debug(f"Unable to parse the interface message: {msg[k]}")

    def _refresh_interface(self, loading_suggestions=False):
        """ Send our entire current state to the frontend interface.
        """
        data = {}

        def create_children(data, tests, topic, include_suggestions):
            children = []

            # add tests and topics to the data lookup structure
            direct_children = tests.get_children_in_topic(topic, direct_children_only=True, include_self=False, include_suggestions=include_suggestions)
            for id, test in direct_children.iterrows():
                # add a topic
                if test.label == "topic_marker":
                    name = test.topic[len(topic)+1:]
                    data[test.topic] = {
                        "label": test.label,
                        "labeler": test.labeler,
                        "description": test.description,
                        "scores": {c: [] for c in self.score_columns},
                        "topic_marker_id": id,
                        "topic_name": name,
                        "editing": test.topic.endswith("/New topic")
                    }
                    # fill in the scores for the child topics using their tests
                    for k, t in tests.get_children_in_topic(
                        test.topic, include_topics=False, direct_children_only=False, include_self=False, include_suggestions=False,
                    ).iterrows():
                        for c in self.score_columns: 
                            data[test.topic]["scores"][c].extend([[k, v] for v in ui_score_parts(t[c], t.label)])
                    children.append(test.topic)
                
                # add a test
                elif self._matches_filter(test, self.filter_text): # apply the filter
                    data[id] = {
                        "input": test.input_display,
                        "output": test.output,
                        "label": test.label,
                        "label_confidence": test.label_confidence,
                        "labeler": test.labeler,
                        "description": test.description,
                        "scores": {c: [[id, v] for v in ui_score_parts(test[c], test.label)] for c in self.score_columns},
                        "editing": test.input == "New test"
                    }
                    data[id]["raw_outputs"] = { # when is this used?
                        c: [[id, adatest.utils.safe_json_load(test.get(c.replace(" score", "") + " raw outputs", "{}"))]] 
                        for c in self.score_columns
                    }
                    data[id].update(FRONTEND_BOILERPLATE) # boilerplate keys needed for frontend
                    children.append(id)
            
            # sort children into display order
            def sort_topic(id):
                total, count = 0, 0
                for s in data[id]["scores"][self.score_columns[0]]:
                    val = adatest.utils.score_max(s[1], nan_val=np.nan)
                    if not np.isnan(val) and val is not None:
                        total += val #+ offset = 0 if data[id]["label"] == "fail" else -1
                        count += 1
                if count == 0:
                    if id.endswith("/" + adatest.utils.sanitize_topic_name("New topic")):
                        return -1e5
                    else:
                        return -1e3 # place new topics first
                else:
                    return -total / count
            
            def sort_test(id):
                # sort first using labeler conf, and then using model score
                key = (
                    adatest.utils.score_max(data[id]["label_confidence"], nan_val=0), # converts to float
                    -adatest.utils.score_max(data[id]["scores"][self.score_columns[0]][0][1], nan_val=0)
                )
                # sort failures from highest to lowest
                if data[id]["label"] == "fail": return (-1 * key[0], key[1])
                # sort pass & off-topic from lowest to highest
                else: return key
            
            topics = sorted(filter(lambda id: data[id].get("label", "") == "topic_marker", children), key=sort_topic) # put folders first
            failed_tests = sorted(filter(lambda id: data[id].get("label", "") == "fail", children), key=sort_test) # put failures next
            passed_tests = sorted(filter(lambda id: data[id].get("label", "") == "pass", children), key=sort_test) # put passed next
            off_topic_tests = sorted(filter(lambda id: data[id].get("label", "") not in ("topic_marker","pass", "fail"), children), key=sort_test) # put Unknown & off_topic last

            sorted_children = topics + failed_tests + passed_tests + off_topic_tests
            return sorted_children
        
        # get the children of the current topic
        children = create_children(data, self.test_tree, self.current_topic, include_suggestions=False)
        suggestions_children = create_children(data, self.test_tree, self.current_topic + "/__suggestions__", include_suggestions=True)

        topic_marker_id = self.test_tree.get_topic_id(self.current_topic)
        # compile the global browser state for the frontend
        data["browser"] = {
            "suggestions": suggestions_children,
            "tests": children,
            "user": self.user,
            "topic": self.current_topic,
            "topic_description": self.test_tree.loc[topic_marker_id]["description"] if topic_marker_id is not None else "",
            "topic_marker_id": topic_marker_id if topic_marker_id is not None else uuid.uuid4().hex,
            "disable_suggestions": False,
            "read_only": False,
            "score_columns": self.score_columns,
            "suggestions_error": self._suggestions_error,
            "generator_options": [str(x) for x in self.generators.keys()] if isinstance(self.generators, dict) else [self.active_generator],
            "active_generator": self.active_generator,
            "loading_suggestions": loading_suggestions, # although ideally this should be tracked by the frontened, there are some edge cases so easier to track in backend
            "mode": self.mode,
            "mode_options": self.mode_options(self.current_topic), 
            "test_tree_name": self.test_tree.name
        }

        self.comm.send(data)

    def _matches_filter(self, test, filter_text):
        """Return if filter_text is a substring of a single test"""
        if filter_text is None or filter_text == "": return True
        elif '__suggestions__' in test.topic: return True # show all suggestions
        else:
            if test["input"].startswith("__IMAGE="):
                return filter_text in test["output"]
            else:
                return filter_text in test["input"] or filter_text in test["output"]

    def _clear_suggestions(self):
        """ Clear the suggestions for the current topic.
        """
        ids = self.test_tree.get_children_in_topic(self.current_topic + "/__suggestions__", include_topics=True, include_suggestions=True, direct_children_only=True).index
        for k in ids:
            try: os.remove(f"{self._static_dir}/{k}.jpg") # remove tmp image
            except: pass
            self.test_tree.drop(k, inplace=True)

    def _change_mode(self, mode):
        """Set the testing mode, and switch to the appropriate generator if a generator has the same name as the mode"""
        self.mode = mode
        if self.mode in self.generators:
            self.active_generator = self.mode
            self._active_generator_obj = self.generators[self.active_generator]

    def _generate_suggestions(self, filter):
        """ Generate suggestions for the current topic.

        Parameters
        ----------
        filter : str
            The filter to apply to the tests while generating suggestions.
        """
        # compute the maximum number of suggestion threads we can use given our suggestion_thread_budget
        p = self.prompt_builder.prompt_size
        budget = 1 + self.suggestion_thread_budget
        suggestion_threads = max(1, int(np.floor(budget * (p/(p+1) + 1/(p+1) * self.max_suggestions) - 1/(p+1) * self.max_suggestions) / (p/(p+1))))
        
        # generate the prompts for the backend
        prompts = self.prompt_builder(
            test_tree=self.test_tree,
            topic=self.current_topic,
            score_column=self.score_columns[0],
            repetitions=suggestion_threads,
            filter=filter,
            suggest_topics=self.mode == "topics"
        )

        # get the current topic description
        curr_topic_mask = (self.test_tree["topic"] == self.current_topic) & (self.test_tree["label"] == "topic_marker")
        if curr_topic_mask.sum() == 0:
            desc = ""
        else:
            desc = self.test_tree.loc[(self.test_tree["topic"] == self.current_topic) & (self.test_tree["label"] == "topic_marker")]["description"][0]

        # generate the suggestions
        generators = [self._active_generator_obj] + list(self.generators.values())
        for generator in generators:
            try:
                # TODO: suggestion thread logic
                num_samples = self.max_suggestions // len(prompts) if len(prompts) > 0 else self.max_suggestions
                proposals = generator(prompts, self.current_topic, desc, self.mode, self.scorer, num_samples=num_samples)
                break
            except ValueError as e:
                print(f"Generator {generator} failed with error {e}. Trying next generator.")
                pass # try the next generator
        
        # all generators failed
        try: proposals
        except: return
        
        if self.mode == "topics":
            proposals = [adatest.utils.sanitize_topic_name(x) for x in proposals]
            # if we generated too many suggestions, subset this down
            if len(proposals) > self.max_suggestions * 2/3:
                proposals = np.random.choice(
                    proposals,
                    int(self.max_suggestions * 2/3),
                    replace=False
                ).tolist()
        
        # postprocess the proposals
        for input in proposals:
            # add suggestion to tree
            id = uuid.uuid4().hex
            self.test_tree.loc[id] = {
                "topic": self.current_topic + "/__suggestions__" + ("/"+input if self.mode == "topics" else ""),
                "input": "" if self.mode == "topics" else input,
                "output": "[no output]",
                "label": "topic_marker" if self.mode == "topics" else "",
                "labeler": "imputed",
                "description": "",
                "author": "suggested",
                "create_time": str(datetime.now()),
            }
            for c in self.score_columns:
                self.test_tree.loc[id, c] = "__TOEVAL__"

        # make sure any duplicates we may have introduced are removed
        if self.mode == "topics":
            num_removed = self.test_tree.deduplicate_subtopics(self.current_topic)
        else:
            num_removed = self.test_tree.deduplicate_tests(self.current_topic)
        
        # if we removed a lot of duplicates, tell the generator, so that it can increase the number of suggestions it tries to generate
        num_remaining = len(proposals) - num_removed
        if num_remaining < 0.75 * self.max_suggestions and hasattr(generator, '_increment_topic_multiple'):
            generator._increment_topic_multiple(self.current_topic)
            if num_remaining == 0: 
                generator._increment_topic_multiple(self.current_topic) # increment again
        
        # compute the scores and topic model labels for the new tests
        self._compute_embeddings_and_scores(self.test_tree)

    def _compute_embeddings_and_scores(self, tests, user_overwrote_output=False):
        """ Use the scorer(s) and topic models to fill in scores and labels in the passed TestTree.
        Examples where this is called: 
            0) When the browser is initialized
            1) After suggestions have just been generated. 
            2) When we add a new test topic. 
            3) When a user modifies the test input from the interface.
            4) When a user modifies the test output from the interface. 
            5) When a user changes a topic (e.g. delete or move).

        Parameters
        ----------
        tests : TestTree
            The TestTree to fill in missing scores for.

        user_overwrote_output : bool
            User changed the output in the interface, so we need to mark score as np.nan
        """
        
        log.debug(f"compute_embeddings_and_scores(tests=<DataFrame shape={tests.shape}>, user_overwrote_output={user_overwrote_output})")

        # nothing to do if we don't have a scorer
        if self.scorer is None:
            return
        
        for k in self.scorer:
            # evaluate all rows that (a) have not been labeled (new suggestions) and don't have a score or output
            # or (b) are on-topic tests and don't have a score or output
            eval_ids = tests.index[((tests[k+" score"] == "__TOEVAL__") | (tests["output"] == "[no output]"))\
                        & (tests["label"] != "topic_marker") \
                        & (tests["label"] != "off_topic")]

            if len(eval_ids) == 0: continue

            # run the scorer
            new_outputs,scores,new_inputs,new_outputs_raw = self.scorer[k](tests, eval_ids)

            # update the scores in the test tree
            current_outputs = tests["output"]
            for i,id in enumerate(eval_ids):
                if user_overwrote_output and current_outputs.loc[id] != new_outputs[i] and current_outputs.loc[id] != "[model output]":
                    # mark the current row as nan score (meaning the output does not match)
                    # if user writes in [model output], we'll let the model take over again
                    tests.loc[id, k+" score"] = np.nan
                    continue
                else:
                    tests.loc[id, "output"] = new_outputs[i]
                    tests.loc[id, k+" score"] = scores[i]
                    
                    if new_outputs_raw is not None: 
                        tests.loc[id, k + " raw output"] = new_outputs_raw[i]
                    
                    if (self._static_dir is not None) and (new_inputs is not None) and (new_inputs[i] is not None): 
                        # if the scorer modified the input (e.g., detection draws in bounding boxes) save the image
                        # and set input_display to the version with the bbox
                        new_inputs[i].save(f"{self._static_dir}/{id}.jpg")
                        tests.loc[id, "input_display"] = f"__IMAGE=/_static/{id}.jpg"
                    else:
                        tests.loc[id, "input_display"] = tests.loc[id, "input"]

        tests.validate_input_displays(self._static_dir)
        tests.get_topic_model_labels()

def ui_score_parts(s, label):
    """ Split a score into its parts and encode the label into the sign.
    Final value will range between [-1, 1], with negative values signifying pass and positive values signifying fail.
    Note this encoding is just used for passing scores to the UI (scores are not signed in the TestTree).
    """
    offset = 0
    if label == "pass":
        sign = -1
        offset = -1e-6 # offset is just to make things more visible in the UI
    elif label == "fail":
        sign = 1
        offset = 1e-6 # just so we have a positive number to encode that this was a failure
    else:
        sign = np.nan
    
    if s is None:
        return [np.nan]

    if isinstance(s, str):
        return [np.clip(offset + adatest.utils.convert_float(v)*sign, -1, 1) for v in s.split("|")]
    else:
        return [np.clip(offset + s*sign, -1, 1)]
