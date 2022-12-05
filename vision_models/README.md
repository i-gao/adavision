## installation
To install all additional requirements used for models provided in this directory, use the following commands:

```python
! pip install openmim # only needed for mmdet models, but must be installed separately
! mim install mmcv-full # only needed for mmdet models, but must be installed separately
! pip install -r vision_models/requirements.txt # all requirements
```

### Amazon Rekognition

If you want to test **Amazon Rekognition**, you'll have to set up an AWS account, activate an API key, and save the API key at a specific, hard-coded location.

See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration
- Visit AWS and create an IAM account.
- In `~/.aws/credentials`, save your AWS access key and AWS secret access key in two lines with appropriate variable names

### Google Cloud Vision
If you want to test **Google Cloud Vision API**, you'll have to set up a GCP project, activate an API, save the API key at a specific, hard-coded location, and change an environmental variable.

See https://codelabs.developers.google.com/codelabs/cloud-vision-api-python#1
- Visit https://console.developers.google.com and create a project.
- Activate the Google Cloud Vision API for your project
- Create a service account
- Save the service account key (JSON) locally and set up an environmental variable `export GOOGLE_APPLICATION_CREDENTIALS=~/key.json`

### Azure Cognitive Services
If you want to test **Azure Cognitive Services**, you'll need an Azure subscription, a Computer Vision resource, and a key.
* Visit https://portal.azure.com/#create/Microsoft.CognitiveServicesComputerVision to create a Computer Vision resource. 
* Within your new resource, go to "Keys and Endpoint" under "Resource Management."
* Save the key and endpoint in a file, where the key goes on the first line, and the endpoint is on the second line. The default path is `~/.azure_key`.

### OFA
To test **One For All (OFA)**, follow these steps: (from https://github.com/OFA-Sys/OFA/issues/171, https://colab.research.google.com/drive/1LLJewY92LXdeug5m_ceMUHdlqrRQwSQJ?usp=sharing#scrollTo=UJdzZAMBZIs4)
1. From within the `vision_models` dir, run the following commands to download the model.
```
git clone --single-branch --branch feature/add_transformers https://github.com/OFA-Sys/OFA.git
pip install OFA/transformers/
```
2. Then, download the finetuned captioning checkpoint:
```
wget https://storage.googleapis.com/mys-released-models/OFA-huge-caption.zip
unzip OFA-huge-caption.zip
```

## loading models
We provide functions to load models from torchvision, timm, mmdet, CLIP, Amazon Rekognition, Google Cloud Vision, Azure Cognitive Services, and OFA. To see all available models, use `vision_models.load.list_models()`.

Models are wrapped in either `vision_models.model.ClassificationModel`, `vision_models.model.DetectionModel`, or `vision_models.model.CaptioningModel`. These wrappers take in a list of image URLs to `__call__()`, convert them to images, batch and transform the images, and format outputs to match the interface expected by AdaVision.