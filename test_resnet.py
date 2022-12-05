from vision_models.load import load_torchvision_model
import adatest
import os

# load the model
model = load_torchvision_model(
    'resnet50', 
    torchvision_weights='ResNet50_Weights.DEFAULT',
)
scorer = adatest.ClassifierScorer(model, top_k=1) 

# start a new tree
tests = adatest.TestTree(f"test_trees/local/my_tree.csv")

# set up the generators
test_generator = adatest.generators.CLIPRetriever(
    aesthetic_weight=0.3
)

with open('prompts.txt') as f:
    prompts = f.readlines()
    prompts = [t.strip() for t in prompts]

topic_generator = adatest.generators.PromptedTopicGenerator(
    values=model.output_names, 
    prompts=prompts, 
    text_completion_generator=adatest.generators.OpenAI(model="text-davinci-002", temperature=0.8, top_p=1)
)

generators = {
    'tests': test_generator, # must put first to use as default
    'topics': topic_generator, 
}

adatest.serve(tests.adapt(
    scorer, generator=generators, max_suggestions=20
), host="0.0.0.0", port=8080)
