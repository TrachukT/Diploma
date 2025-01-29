import json
from urllib.parse import urlparse

import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
from NN_class import ConvNeuralNet  # Import your model class

DETECTION_TYPE = "epidermology"
RESULTS_FOLDER = "results"

#To run this app:
# fastapi dev main.py

app = FastAPI()

class ValidationRequestModel(BaseModel):
    url: str= Field(..., description="Parameter to provide url for image scraping.")

transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),         
    transforms.Normalize(mean=[0.5], std=[0.5])  
])

@app.post("/validate-skin")
async def validate_skin(request: ValidationRequestModel):
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

class ClassificationRequestModel(BaseModel):
    url: str= Field(..., description="Parameter to provide url for image scraping.")
    user_id: str= Field(..., description="Parameter to provide a user identifier.")
    timestamp: str= Field(..., description="Parameter to provide a timestamp of request.")

@app.post("/classify-skin")
async def classify_skin(request: ClassificationRequestModel):
    url = request.url
    user_id = request.user_id
    timestamp = request.timestamp

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
        base_path, old_folder, file_name = url.rsplit("/", 2)
        new_file_name = f"{user_id}_{DETECTION_TYPE}_{timestamp}.txt"
        new_s3_path = f"{base_path}/{RESULTS_FOLDER}/{new_file_name}"
        s3_key = "/".join(new_s3_path.split("/")[3:])

        s3.put_object(Bucket=bucket_name, Key=s3_key, Body=json.dumps({**result,"image_url":url}),
                      ContentType="application/json")
        return {**result, "path": new_s3_path}


    except requests.RequestException as e:
        raise HTTPException(status_code=response.status_code, detail=f"Error fetching image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
