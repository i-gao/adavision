""" A set of generators for AdaTest.
"""
import asyncio
import aiohttp
from profanity import profanity
import numpy as np
import os
import re
import adatest
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

class Generator():
    """ Abstract class for generators.
    """

    def __init__(self, source):
        """ Create a new generator from a given source.
        """
        self.source = source

    def __call__(self, prompts, topic, topic_description, mode, scorer, num_samples, max_length):
        """ Generate a set of prompts for a given topic.
        """
        raise NotImplementedError()

    def _validate_prompts(self, prompts):
        """ Ensure the passed prompts are a list of prompt lists.
        """
        if len(prompts[0]) > 0 and isinstance(prompts[0][0], str):
            prompts = [prompts]
        
        # Split apart prompt IDs and prompts
        prompt_ids, trimmed_prompts = [], []
        for prompt in prompts:
            prompt_without_id = []
            for entry in prompt:
                prompt_ids.append(entry[0])
                prompt_without_id.append(entry[1:])
            trimmed_prompts.append(prompt_without_id)

        return trimmed_prompts, prompt_ids


class TextCompletionGenerator(Generator):
    """ Abstract class for generators.
    """
    
    def __init__(self, source, sep, subsep, quote, filter):
        """ Create a new generator with the given separators and max length.
        """
        super().__init__(source)
        self.sep = sep
        self.subsep = subsep
        self.quote = quote
        self.filter = filter

    def __call__(self, prompts, topic, topic_description, max_length=None):
        """ This should be overridden by concrete subclasses.

        Parameters:
        -----------
        prompts: list of tuples, or list of lists of tuples
        """
        pass
    
    def _direct_children(self, prompts, topic):
        """ Returns True if all prompts are direct children of the current topic 
        """
        for prompt in prompts:
            if len(prompt) == 0: continue
            topics, inputs = zip(*prompt)
            if len(set(list(topics) + [topic])) > 1:
                return False
        return True
    
    def _create_prompt_strings(self, prompts, topic, content_type):
        """ Convert prompts that are lists of tuples into strings for the LM to complete.
        """

        assert content_type in ["tests", "topics"], "Invalid mode: {}".format(content_type)

        show_topics = not self._direct_children(prompts, topic) or content_type == "topic"

        prompt_strings = []
        for prompt in prompts:
            prompt_string = ""
            for p_topic, input in prompt:
                if show_topics:
                    if content_type == "tests":
                        prompt_string += self.sep + p_topic + ":" + self.sep + self.quote
                    elif content_type == "topics":
                        prompt_string += "A subtopic of " + self.quote + p_topic + self.quote + " is " + self.quote
                else:
                    prompt_string += self.quote
                
                prompt_string += input + self.quote
                prompt_string += self.sep
            if show_topics:
                if content_type == "tests":
                    prompt_strings.append(prompt_string + self.sep + topic + ":" + self.sep + self.quote)
                elif content_type == "topics":
                    prompt_strings.append(prompt_string + "A subtopic of " + self.quote + topic + self.quote + " is " + self.quote)
            else:
                prompt_strings.append(prompt_string + self.quote)
        return prompt_strings
    
    def _parse_suggestion_texts(self, suggestion_texts, prompts):
        """ Parse the suggestion texts into tuples.
        """
        assert len(suggestion_texts) % len(prompts) == 0, "Missing prompt completions!"

        # _, gen_value1, gen_value2, gen_value3 = self._varying_values(prompts, "") # note that "" is an unused topic argument
        
        num_samples = len(suggestion_texts) // len(prompts)
        samples = []
        for i, suggestion_text in enumerate(suggestion_texts):
            if callable(self.filter):
                suggestion_text = self.filter(suggestion_text)
            samples.append(suggestion_text)
        return list(set(samples))

class HuggingFace(TextCompletionGenerator):
    """This class exists to embed the StopAtSequence class."""
    import transformers

    def __init__(self, source, sep, subsep, quote, filter):
        super().__init__(source, sep, subsep, quote, filter)
        

    class StopAtSequence(transformers.StoppingCriteria):
        def __init__(self, stop_string, tokenizer, window_size=10):
            self.stop_string = stop_string
            self.tokenizer = tokenizer
            self.window_size = 10
            self.max_length = None
            self.prompt_length = 0
            
        def __call__(self, input_ids, scores):
            if len(input_ids[0]) > self.max_length + self.prompt_length:
                return True

            # we need to decode rather than check the ids directly because the stop_string may get enocded differently in different contexts
            return self.tokenizer.decode(input_ids[0][-self.window_size:])[-len(self.stop_string):] == self.stop_string

           
class Transformers(HuggingFace):
    def __init__(self, model, tokenizer, sep="\n", subsep=" ", quote="\"", filter=profanity.censor):
        # TODO [Harsha]: Add validation logic to make sure model is of supported type.
        super().__init__(model, sep, subsep, quote, filter)
        self.gen_type = "model"
        self.tokenizer = tokenizer
        self.device = self.source.device

        self._sep_stopper = HuggingFace.StopAtSequence(self.quote+self.sep, self.tokenizer)
    
    def __call__(self, prompts, topic, topic_description, mode, scorer=None, num_samples=1, max_length=100):
        prompts, prompt_ids = self._validate_prompts(prompts)
        if len(prompts) == 0:
            raise ValueError("ValueError: Unable to generate suggestions from completely empty TestTree. Consider writing a few manual tests before generating suggestions.") 
        prompt_strings = self._create_prompt_strings(prompts, topic, mode)
        
        # monkey-patch a method that prevents the use of past_key_values
        saved_func = self.source.prepare_inputs_for_generation
        def prepare_inputs_for_generation(input_ids, **kwargs):
            if "past_key_values" in kwargs:
                return {"input_ids": input_ids, "past_key_values": kwargs["past_key_values"]}
            else:
                return {"input_ids": input_ids}
        self.source.prepare_inputs_for_generation = prepare_inputs_for_generation
        
        suggestion_texts = self.sample_suggestions(prompt_strings=prompt_strings, num_samples_per_prompt=num_samples, max_length=max_length)

        # restore the old function that prevents the past_key_values argument from getting passed
        self.source.prepare_inputs_for_generation = saved_func
        
        return self._parse_suggestion_texts(suggestion_texts, prompts)
    
    def sample_suggestions(self, prompt_strings, num_samples_per_prompt, max_length=100):
        # run the generative LM for each prompt
        suggestion_texts = []
        for prompt_string in prompt_strings:
            input_ids = self.tokenizer.encode(prompt_string, return_tensors='pt').to(self.device)
            cache_out = self.source(input_ids[:, :-1], use_cache=True)

            for _ in range(num_samples_per_prompt):
                self._sep_stopper.prompt_length = 1
                self._sep_stopper.max_length = max_length
                out = self.source.sample(
                    input_ids[:, -1:], pad_token_id=self.source.config.eos_token_id,
                    stopping_criteria=self._sep_stopper,
                    past_key_values=cache_out.past_key_values # TODO: enable re-using during sample unrolling as well
                )

                # we ignore first token because it is part of the prompt
                suggestion_text = self.tokenizer.decode(out[0][1:])
                
                # we ignore the stop string to match other backends
                if suggestion_text[-len(self._sep_stopper.stop_string):] == self._sep_stopper.stop_string:
                    suggestion_text = suggestion_text[:-len(self._sep_stopper.stop_string)]
                
                suggestion_texts.append(suggestion_text)
        return suggestion_texts


class Pipelines(HuggingFace):
    import transformers
    def __init__(self, pipeline: transformers.pipelines.base.Pipeline , sep="\n", subsep=" ", quote="\"", filter=profanity.censor):
        super().__init__(pipeline, sep, subsep, quote, filter)
        self.gen_type = "model"
        self.stop_sequence = self.quote + self.sep
        self._sep_stopper = HuggingFace.StopAtSequence(self.stop_sequence, pipeline.tokenizer)

    def __call__(self, prompts, topic, topic_description, mode, scorer=None, num_samples=1, max_length=100):
        if len(prompts) == 0:
            raise ValueError("ValueError: Unable to generate suggestions from completely empty TestTree. Consider writing a few manual tests before generating suggestions.") 
        prompts, prompt_ids = self._validate_prompts(prompts)
        prompt_strings = self._create_prompt_strings(prompts, topic, mode)

        suggestion_texts = self.sample_suggestions(prompt_strings=prompt_strings, num_samples_per_prompt=num_samples, max_length=max_length)

        return self._parse_suggestion_texts(suggestion_texts, prompts)

    def sample_suggestions(self, prompt_strings, num_samples_per_prompt, max_length=100):
        suggestion_texts = []
        for p in prompt_strings:
            prompt_length = len(self.source.tokenizer.tokenize(p))
            self._sep_stopper.prompt_length = prompt_length
            self._sep_stopper.max_length = max_length
            generations = self.source(p,
                        do_sample=True,
                        max_length=prompt_length + max_length,
                        num_return_sequences=num_samples_per_prompt,
                        pad_token_id=self.source.model.config.eos_token_id,
                        stopping_criteria=[self._sep_stopper])
            for gen in generations:
                generated_text = gen['generated_text'][len(p):]
                # Trim off text after stop_sequence
                stop_seq_index = generated_text.find(self.stop_sequence)
                if (stop_seq_index != -1):
                    generated_text = generated_text[:stop_seq_index]
                elif generated_text[-1] == self.quote:
                    # Sometimes the quote is at the end without a trailing newline
                    generated_text = generated_text[:-1]
                suggestion_texts.append(generated_text)
        return suggestion_texts


class OpenAI(TextCompletionGenerator):
    """ Backend wrapper for the OpenAI API that exposes GPT-3.
    """
    
    def __init__(self, model="curie", api_key=None, sep="\n", subsep=" ", quote="\"", temperature=1.0, top_p=0.95, filter=profanity.censor):
        import openai

        # TODO [Harsha]: Add validation logic to make sure model is of supported type.
        super().__init__(model, sep, subsep, quote, filter)
        self.gen_type = "model"
        self.temperature = temperature
        self.top_p = top_p
        if api_key is not None:
            openai.api_key = api_key

        # load a key by default if a standard file exists
        elif openai.api_key is None:
            key_path = os.path.expanduser("~/.openai_api_key")
            if os.path.exists(key_path):
                with open(key_path) as f:
                    openai.api_key = f.read().strip()

    def __call__(self, prompts, topic, topic_description, mode, scorer=None, num_samples=1, max_length=100):
        if len(prompts[0]) == 0:
            raise ValueError("ValueError: Unable to generate suggestions from completely empty TestTree. Consider writing a few manual tests before generating suggestions.") 

        prompts, prompt_ids = self._validate_prompts(prompts)

        # create prompts to generate the model input parameters of the tests
        prompt_strings = self._create_prompt_strings(prompts, topic, mode)
        
        suggestion_texts = self.sample_suggestions(prompt_strings=prompt_strings, num_samples_per_prompt=num_samples, max_length=max_length)
        
        return self._parse_suggestion_texts(suggestion_texts, prompts)

    def sample_suggestions(self, prompt_strings, num_samples_per_prompt, max_length=100):
        # call the OpenAI API to complete the prompts
        import openai

        response = openai.Completion.create(
            model=self.source, prompt=prompt_strings, max_tokens=max_length, user="adatest",
            temperature=self.temperature, top_p=self.top_p, n=num_samples_per_prompt, stop=self.quote
        )
        suggestion_texts = [choice["text"] for choice in response["choices"]]
        return suggestion_texts


class AI21(TextCompletionGenerator):
    """ Backend wrapper for the AI21 API.
    """
    
    def __init__(self, model, api_key, sep="\n", subsep=" ", quote="\"", temperature=0.95, filter=profanity.censor):
        # TODO [Harsha]: Add validation logic to make sure model is of supported type.
        super().__init__(model, sep, subsep, quote, filter)
        self.gen_type = "model"
        self.api_key = api_key
        self.temperature = temperature
        self.event_loop = asyncio.get_event_loop()
    
    def __call__(self, prompts, topic, topic_description, mode, scorer=None, num_samples=1, max_length=100):
        prompts, prompt_ids = self._validate_prompts(prompts)
        prompt_strings = self._create_prompt_strings(prompts, topic, mode)
        
        suggestion_texts = self.sample_suggestions(prompt_strings=prompt_strings, num_samples_per_prompt=num_samples, max_length=max_length)
        
        return self._parse_suggestion_texts(suggestion_texts, prompts)
    
    def sample_suggestions(self, prompt_strings, num_samples_per_prompt, max_length=100):
        # define an async call to the API
        async def http_call(prompt_string):
            async with aiohttp.ClientSession() as session:
                async with session.post(f"https://api.ai21.com/studio/v1/{self.source}/complete",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "prompt": prompt_string, 
                            "numResults": num_samples_per_prompt, 
                            "maxTokens": max_length,
                            "stopSequences": [self.quote+self.sep],
                            "topKReturn": 0,
                            "temperature": self.temperature
                        }) as resp:
                    result = await resp.json()
                    return [c["data"]["text"] for c in result["completions"]]
        
        # call the AI21 API asyncronously to complete the prompts
        results = self.event_loop.run_until_complete(asyncio.gather(*[http_call(s) for s in prompt_strings]))
        suggestion_texts = []
        for result in results:
            suggestion_texts.extend(result)
        return suggestion_texts

class CLIPRetriever(Generator):
    """
    Image retrieval system built on top of the clip-retrieval python package (https://github.com/rom1504/clip-retrieval/)
    and using the provided live KNN backend.

    The clip-retrieval package is build on top of the Laion5B project (https://laion.ai/blog/laion-5b/),
    which involved curating a 5B (image, text) dataset, computing CLIP ViT-L/14 embeddings for all examples, 
    and converting embeddings into a PQ128 KKN index. The clip-retrieval project provides python infrastructure 
    for a KNN backend based on Laion5B indices and a frontend client that makes requests to the backend.

    We directly use clip-retrieval's live backend (https://knn5.laion.ai/knn-service) and use the package's 
    frontend client to query this backend.
    """
    def __init__(self, 
                 use_full_laion=True, 
                 aesthetic_weight=0, 
                 default_num_images=30, 
                 use_safety_model=True, 
                 use_violence_detector=True,
                 default_num_augs=2,
                 use_text_only=False,
                 text_prefix="",
                 multiple_increment=5,
    ):
        """
        Args:
            - use_full_laion (bool) -- whether to use Laion5B over Laion400M
            - aesthetic_weight (float) -- aesthetic weight ot use for clip-retrieval
            - default_num_images (int) -- default num neighbors to query for
            - use_safety_model (bool) -- whether to ask clip-retrieval to filter for "safe" images
            - use_violence_detector (bool) -- whether to ask clip-retrieval to filter for non-"violent" images
            - default_num_augs (int) -- number of times to augment the query vector using the embedding_augmentation function
            - use_text_only (bool) -- baseline condition: don't use image prompts to retrieve; only use text
        """
        from clip_retrieval.clip_client import ClipClient, Modality

        index_name = 'laion5B' if use_full_laion else 'laion_400m'
        super().__init__(index_name)
        self.engine = ClipClient(
            url="https://knn5.laion.ai/knn-service",
            indice_name=index_name,
            modality=Modality.IMAGE,
            deduplicate=True,
            aesthetic_weight=aesthetic_weight,
            num_images=default_num_images,
            use_safety_model=use_safety_model,
            use_violence_detector=use_violence_detector,
        )
        self.default_num_images = default_num_images
        self.num_augs = default_num_augs
        self.use_text_only = use_text_only
        self.prefix = text_prefix

        # An annoying property of CLIP Retrieval is that they sometimes return << number of images that you request
        # To make this less obvious to the user, keep track of a multiplicative factor for each topic that we should
        # multiply num_samples by. The default is set to 10.
        self.topic_multiples = defaultdict(lambda:10)
        self.multiple_increment = multiple_increment

    def embedding_aggregation(self, txts: list, imgs: list):
        """
        Given a list of text and image embeddings, return a single embedding to query with.
        Take a random weighted mean of the text vectors, and a random weighted mean of the image vectors, 
        and then call slerp to interpolate between the two weighted means with a random parameter t.
        With probability 0.25, only use the text embedding.
        Otherwise, upweight the image embedding weight by 0.25.
        """
        # weighted mean of text embeddings
        if len(txts):
            txt_weights = np.random.dirichlet(np.ones(len(txts)))
            txts = np.average(txts, axis=0, weights=txt_weights)
        else:
            txts = None
        
        # weighted mean of image embeddings
        if len(imgs):
            img_weights = np.random.dirichlet(np.ones(len(imgs)))
            print(img_weights)
            imgs = np.average(imgs, axis=0, weights=img_weights)
        else:
            imgs = None
        
        assert not (txts is None and imgs is None), "No vectors were passed in to aggregate!"

        # interpolate between the two weighted means
        t = np.random.rand()
        if txts is None:
            return imgs
        elif imgs is None or (t < 0.25 and txts is not None):
            return txts
        else:
            print("Calling slerp with t=", min(t + 0.25, 1))
            return adatest.utils.slerp(txts, imgs, t=min(t + 0.25, 1))

    def embedding_augmentation(self, x):
        """Given an embedding, add a bit of gaussian noise and return the augmented embedding."""
        import scipy.stats as stats
        return x + stats.norm.rvs(loc=0, scale=0.01, size=len(x))

    def __call__(self, prompts, topic, topic_description, mode, scorer=None, num_samples=1, max_length=100):
        """ Generate suggestions for the given topic and prompts.
        """
        if mode == 'topics':
            raise ValueError("CLIPRetriever fetches images, not text, so it can't be used for generating topics." \
            + "Try including both adatest.CLIPRetriever and adatest.OpenAI in the generators argument of adapt() as a dict.")

        # make sure we have valid prompts
        prompts, prompt_ids = self._validate_prompts(prompts)
        
        # use the topic and (parent topics) as the query
        # NOTE: we do NOT use the description, which is meant for the user to make notes to themself about the kinds of bugs they encounter.
        text_query = topic.replace("%20", " ").split("/")[-1]
        # text_query = " " .join(topic.replace("%20", " ").split("/")[::-1])
        print(text_query)

        # embed the images in the prompts
        suggestion_texts = []
        for p in prompts:
            urls = [v[1] for v in p]

            if len(urls) == 0 and text_query == "":
                raise ValueError("ValueError: Unable to generate suggestions from completely empty TestTree and no topic description. Consider adding some topics (or a topic description) before generating test suggestions.")

            response = self.query(
                text=self.prefix + text_query,
                urlpath=urls if not self.use_text_only else None,
                num_samples=num_samples,
                topic=topic,
            )
            suggestion_texts.extend(["__IMAGE="+result["url"] for result in response])

        # return a unique list of suggestions
        return list(set(suggestion_texts))
    
    def query(self, text=None, urlpath=None, num_samples=None, topic=""):
        """
        Wraps ClipClient query() function. Depending on the download_images flag, is capable of outputting:
            query() -> [{'caption': TEXT, 'url': IMAGE URL, 'id': INT, 'similarity': FLOAT}]
        Can take in several object(s) to use as the KNN query, including text(s) and image URL(s).
        Args:
            Query objects
            - text (str or [str]) -- text(s) to query by
            - urlpath (str or [urlpath]) -- url(s) of image to query by
        Returns:
            - list of dicts, where dicts have keys 'caption', 'url', 'id', 'similarity'
                len of list is at most num_samples
        """
        # compute num_images to request per query 
        # we'll return <= num_samples results, aiming for num_samples
        # b/c the clip-retrieval API is finnicky, this might require requesting > num_samples NNs
        # note we're also using num_augs query augmentations for diversity, but there may be duplicates between queries
        # it's faster to overstate what we need & then trim down, since we can make requests in parallel
        if num_samples is None: num_samples = self.default_num_images
        multiple = self.topic_multiples[topic]
        self.engine.num_images = int((num_samples // (self.num_augs + 1)) * multiple)
        print(multiple, self.engine.num_images, num_samples)
 
        # get query embedding vectors
        if type(text) != list: text = [text]
        with adatest.utils.Stopwatch() as ET:
            text_embeddings = adatest.embed(text, embedder_name="adatest.embedders.CLIPEmbedding(ViT-L/14)") # defaults to Transformers
            img_embeddings = adatest.embed(urlpath, embedder_name="adatest.embedders.CLIPEmbedding(ViT-L/14)") # defaults to CLIP
            in_embeddings = text_embeddings + img_embeddings # list concatenation

            # aggregate these embeddings into one aggregated vector
            if self.embedding_aggregation is not None:
                qs = [self.embedding_aggregation(text_embeddings, img_embeddings)]
            else:
                qs = in_embeddings
            
            # pass each vector through the retriever's embedding augmentation
            if self.embedding_augmentation is not None:
                to_augment = qs.copy()
                for q in to_augment: qs.extend([self.embedding_augmentation(q) for _ in range(self.num_augs)])
        print(f"Embedded queries in {ET.time}s.")
        
        # submit parallel query requests for eahch query vector; pool results
        _query_wrapper = lambda q: self.engine.query(embedding_input=q.tolist())
        with adatest.utils.Stopwatch() as QT:
            out = []
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(_query_wrapper, q) for q in qs]
            for f in futures: out.extend(f.result())
        print(f"Queried KNN in {QT.time}s.")

        # if KNN gave us fewer examples than we requested, increase the topic multiple
        # if len(out) < num_samples:
        #     self._increment_topic_multiple(topic)

        # in case we get too many results, truncate down to <= multiple * num_samples results for speed
        out = [out[i] for i in np.random.choice(len(out), size=min(len(out), num_samples * multiple), replace=False)]
        return out    

    def _increment_topic_multiple(self, topic):
        self.topic_multiples[topic] += self.multiple_increment


class GoogleRetriever(Generator):
    def __init__(self, 
                 default_num_images=30, 
                 safe_search='active',
                 cc_rights='cc_publicdomain',
                 api_key_path='~/.google_custom_search_key',
                 multiple_increment=5,
    ):
        """
        Google Image search. Built on top of the Google-Images-Search package (https://github.com/arrrlo/Google-Images-Search). 
        
        Set up:
            - Visit https://console.developers.google.com and create a project.
            - Visit https://console.developers.google.com/apis/library/customsearch.googleapis.com and enable "Custom Search API" for your project.
            - Visit https://console.developers.google.com/apis/credentials and generate API key credentials for your project.
            - Visit https://cse.google.com/cse/all and create a custom search engine to generate a Custom Search Engine ID.
                Enable the "Image search" option. 
            - In a file, put your API key as the first line and the Custom Search Engine ID as the second line.

        Args: 
            - api_key_path (path): first line should be the Google Developer API key. second line is the custom search engine ID. 
        """
        from google_images_search import GoogleImagesSearch

        super().__init__("google")
        with open(api_key_path, 'r') as f: 
            API_KEY = f.readline().replace('\n', '')
            CSE_ID = f.readline().replace('\n', '')
        self.engine = GoogleImagesSearch(API_KEY, CSE_ID)
        self.default_search_params = {
            'safe': safe_search,
            'rights': cc_rights,
            'num': default_num_images,
        }
        
        # An annoying property of Google is that it's non-stochatic; after a few fetches, we get duplicate results.
        # To make this less obvious to the user, keep track of a multiplicative factor for each topic that we should
        # multiply num_samples by. The default is set to 5.
        self.topic_multiples = defaultdict(lambda:5)
        self.multiple_increment = multiple_increment

    def __call__(self, prompts, topic, topic_description, mode, scorer=None, num_samples=1, max_length=100):
        """ Generate suggestions for the given topic and prompts.
        """
        if mode == 'topics':
            raise ValueError("GoogleRetriever fetches images, not text, so it can't be used for generating topics." \
            + "Try including both adatest.GoogleRetriever and adatest.OpenAI in the generators argument of adapt() as a dict.")

        # prompts are not used, so ignore them
        text_query = topic.replace("%20", " ").split("/")[-1]
        # text_query = " " .join(topic.replace("%20", " ").split("/")[::-1])
        print(text_query)

        if text_query == "":
            raise ValueError("ValueError: Unable to generate suggestions from completely empty TestTree and no topics. Consider adding some topics (or a topic description) before generating test suggestions.")

        # handle multiples
        multiple = self.topic_multiples[topic]

        response = self.query(
            text=text_query,
            num_samples=num_samples,
        )
        suggestion_texts = ["__IMAGE="+result["url"] for result in response]

        # return a unique list of suggestions
        return list(set(suggestion_texts))

    def _increment_topic_multiple(self, topic):
        self.topic_multiples[topic] += self.multiple_increment

    def query(self, text: str, num_samples=None):
        """
        Wraps Google Image Search package. Depending on the download_images flag, is capable of outputting:
            query() -> [{'url': IMAGE URL, 'img': PIL IMAGE}]
        Can only take in a single string of text to use as the KNN query.
        """
        # set up search params
        search_params = {
            **self.default_search_params,
            'q': text,
        }
        if num_samples is not None: search_params['num'] = num_samples
        
        # submit request
        with adatest.utils.Stopwatch() as QT:
            self.engine._search_again = False
            self.engine.search(search_params=search_params)
            out = [{'url': image.url} for image in self.engine.results()]
        print(f"Queried Google in {QT.time}s and got {len(out)} results.")
        return out    
        
class ConstantGenerator(Generator):
    """ Generator that returns a constant set of strings (e.g., class names)
    """
    def __init__(self, values):
        assert type(values[0] ) == str
        super().__init__(values)
        self.values = set(values)

    def __call__(self, prompts, topic, topic_description, mode, scorer=None, num_samples=1, max_length=100):
        """ Generate a set of prompts for a given topic.
        Essentially deduplicates self.values against prompts and topic, and then calls it a day.
        """
        # make sure we have valid prompts
        prompts, prompt_ids = self._validate_prompts(prompts)

        # deduplicate against existing topics at the same level
        existing_topics = set([topic]) 
        for p in prompts:
            existing_topics = existing_topics.union(set([v[1] for v in p if v[0].startswith(topic)]))

        options = list(self.values - existing_topics)
        return np.random.choice(
            options,
            size=min(num_samples, len(options)),
            replace=False
        ).tolist()

class PromptedTopicGenerator(Generator):
    """
    Use a text generator to modify a fixed set of values
    TODO: document this better
    """
    def __init__(self, values, prompts, text_completion_generator, enforce_values_in_root_suggestions=True, enforce_values_in_subtopic_suggestions=True):
        assert type(values[0] ) == str and type(prompts[0]) == str
        super().__init__(prompts)
        self.values = list(set(values))
        self.templates = [tuple(p.split('|')) for p in list(set(prompts))] # templates expected have an input and output format, separated by |
        self.engine = text_completion_generator # e.g., OpenAI
        self.enforce_values_in_root_suggestions = enforce_values_in_root_suggestions
        self.enforce_values_in_subtopic_suggestions = enforce_values_in_subtopic_suggestions

    def __call__(self, prompts, topic, topic_description, mode, scorer=None, num_samples=1, max_length=10):
        """ TODO
        """        
        if mode != 'topics':
            raise ValueError("This generator can only be used to generate text for topics.")

        # generate prompt strings (independently of passed-in prompts)
        # we'll aim to get num_samples // 2 with this first round of prompting
        prompt_strings, template_ixs, chosen_values = self._create_prompt_strings(num_samples // 10, num_values=1, topic=topic.replace("%20", " ").split("/")[-1])
        print(prompt_strings)
        
        # call the text engine to complete the prompts
        suggestion_texts = self.engine.sample_suggestions(prompt_strings=prompt_strings, num_samples_per_prompt=1, max_length=100)
        
        # format responses in the template's out format
        suggestion_texts = self._parse_suggestion_texts(suggestion_texts, template_ixs, chosen_values)

        # prompt GPT again naively with both the prompts and the suggestions
        # aim to get the the other half of num_samples using this naive round
        new_prompts = [("", topic, s) for s in suggestion_texts]
        prompts = [new_prompts + p for p in prompts]
        try:
            new_suggestion_texts = self.engine(
                prompts, topic, topic_description, mode, num_samples=num_samples//2
            )
            print(f"Naive GPT added {len(new_suggestion_texts)} suggestions: {new_suggestion_texts}")
            suggestion_texts = suggestion_texts + new_suggestion_texts
        except:
            pass

        # filter out suggestions which have no mention of a value
        suggestion_texts = list(filter(
            lambda s: (
                len(s.split(" ")) <= max_length) \
                and (not (
                    (self.enforce_values_in_root_suggestions and topic == "")
                    or (self.enforce_values_in_subtopic_suggestions and topic != "")
                ) or np.any([adatest.utils.mostly_substring(value, s) for value in chosen_values])
            ),
            suggestion_texts
        ))
        return [s.lower() for s in suggestion_texts]

    def _create_prompt_strings(self, num_samples, num_values=1, topic=""):
        """
        TODO: better documentation
        Generate num_samples prompts about num_values values from our handwritten prompts loaded in __init__
        Returns:
            - prompt_strings (list of len num_samples) formated prompts about num_values values
            - template_ixs (list of len num_samples) self.templates[ix] is the template that prompt_string[ix] was generated from
            - values_used (list of len num_samples) values_used[ix] is the value that prompt_string[ix] is about
        """
        # randomly select some prompts and format them
        template_ixs = np.random.choice(
            len(self.templates),
            size=num_samples,
            replace=(len(self.templates) < num_samples)
        ).tolist()
        
        # assume that each template only asks for one value ({} will be replaced with the same value)
        if topic != "":
            value_options = [topic] # if asking for subtopics, use the current topic as the value
        else:
            value_options = np.random.choice(
                self.values,
                size=num_values,
                replace=(len(self.values) < num_values)
            ).tolist()
        
        prompt_strings, values_used = [], []
        for i, ix in enumerate(template_ixs):
            num_slots = self.templates[ix][0].count("{}")
            value_to_use = value_options[i % num_values]
            input = self.templates[ix][0].format(*([value_to_use] * num_slots))
            input = input.replace("\\n", "\n")
            prompt_strings.append(input)
            values_used.append(value_to_use)
        
        return prompt_strings, template_ixs, values_used

    def _parse_suggestion_texts(self, suggestion_texts, template_ixs, values):
        """ Parse the suggestion texts into tuples.
        """
        assert len(suggestion_texts) == len(template_ixs), "Missing suggestions!"

        outputs = []
        for i in range(len(suggestion_texts)):
            # format output template with the original value picked in the prompt
            template = self.templates[template_ixs[i]][1]
            num_values = template.count("{}")
            template = template.format(*([values[i]] * num_values))
            template = template.replace('[]', '{}')
            num_values = template.count("{}")

            # filter the suggestions as a group, and then...
            suggestion_text = suggestion_texts[i]
            if callable(self.engine.filter):
                suggestion_text = self.engine.filter(suggestion_text)

            # plug in each suggestion to generate an output
            suggestion_text = suggestion_text.replace("\\n", "\n")
            for line in re.split("[\n(\d+\))]", suggestion_text)[1:-1]: # split on newline or 1); skip last sugg, which may be cut off
                # validate line; extract the completion from the line
                line = re.split("[.!?]", line)[0]
                line = re.sub(r'[^\'\-\w\s]', '', line)
                if len(line.strip()) == 0: continue
                
                # fill in the template and append
                output = template.format(*([line.strip()] * num_values))
                output = output.replace(f'{values[i]} {values[i]}', values[i]) # trim down duplicates
                outputs.append(output)

        return list(set(outputs))
