from urllib.parse import urlparse

import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
from NN_class import ConvNeuralNet  # Import your model class

#To run this app:
# fastapi dev main.py

app = FastAPI()

class RequestBodyModel(BaseModel):
    url: str

transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),         
    transforms.Normalize(mean=[0.5], std=[0.5])  
])

@app.post("/validate-skin")
async def validate_skin(request: RequestBodyModel):
    url = request.url
    print(url)

    try:
        validation_model = ConvNeuralNet(num_classes=2)
        validation_model.load_state_dict(torch.load("model.pth"))
        validation_model.eval()
        parsed_url = urlparse(url)
        if parsed_url.scheme != "s3":
            raise ValueError("URL повинен починатися з s3://")

        bucket_name = parsed_url.netloc
        object_key = parsed_url.path.lstrip("/")
        s3 = boto3.client('s3')

        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        print(f"Status Code: {response['ResponseMetadata']['HTTPStatusCode']}")
        print(f"Headers: {response['ResponseMetadata']}")

        file_content = response['Body'].read()

        img = Image.open(BytesIO(file_content))
        img_tensor = transform(img).unsqueeze(0)
        
        output = validation_model(img_tensor)
        _, predicted = torch.max(output, 1)
        
        is_skin = predicted.item() == 1  # 1 is 'skin', 0 is 'not skin'
        
        return {"value": is_skin}
    
    except requests.RequestException as e:
        raise HTTPException(status_code=response.status_code, detail=f"Error fetching image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify-skin")
async def classify_skin(request: RequestBodyModel):
    url = request.url

    try:
        classification_model = ConvNeuralNet(num_classes=7)
        classification_model.load_state_dict(torch.load("classification_model.pth"))
        classification_model.eval()

        parsed_url = urlparse(url)
        if parsed_url.scheme != "s3":
            raise ValueError("URL повинен починатися з s3://")

        bucket_name = parsed_url.netloc
        object_key = parsed_url.path.lstrip("/")
        s3 = boto3.client('s3')

        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        print(f"Status Code: {response['ResponseMetadata']['HTTPStatusCode']}")
        print(f"Headers: {response['ResponseMetadata']}")

        file_content = response['Body'].read()

        img = Image.open(BytesIO(file_content))
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = classification_model(img_tensor)
            probabilities = torch.softmax(output, dim=1).squeeze().tolist()

        class_labels = {
            0: "Eczema",
            1: "Warts Molluscum and other Viral Infections",
            2: "Atopic Dermatitis",
            3: "Basal Cell Carcinoma",
            4: "Benign Keratosis-like Lesions",
            5: "Psoriasis pictures Lichen Planus and related diseases",
            6: "Seborrheic Keratoses and other Benign Tumors"
        }

        result = {class_labels[i]: prob for i, prob in enumerate(probabilities)}

        return result


    except requests.RequestException as e:
        raise HTTPException(status_code=response.status_code, detail=f"Error fetching image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
