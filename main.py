import json
from urllib.parse import urlparse

import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
import torch.nn as nn
from io import BytesIO
from models.training.NN_class import ConvNeuralNet  # Import your model class
from models.training.utils import train, evaluate
from config import CLASS_LABELS, DETECTION_TYPE, RESULTS_FOLDER

# To run this app:
# fastapi dev main.py

app = FastAPI()


class S3ClientSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = boto3.client("s3")
        return cls._instance


class ValidationRequestModel(BaseModel):
    url: str = Field(..., description="Parameter to provide url for image scraping.")


transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

validation_model = ConvNeuralNet(num_classes=2)
validation_model.load_state_dict(torch.load("models/files/model.pth"))
validation_model.eval()

classification_model = ConvNeuralNet(num_classes=7)
classification_model.load_state_dict(torch.load("models/files/classification_model.pth"))
classification_model.eval()


@app.post("/validate-skin")
async def validate_skin(request: ValidationRequestModel):
    url = request.url
    print(url)

    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme != "s3":
            raise ValueError("URL повинен починатися з s3://")

        bucket_name = parsed_url.netloc
        object_key = parsed_url.path.lstrip("/")
        s3 = S3ClientSingleton()

        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        print(f"Status Code: {response['ResponseMetadata']['HTTPStatusCode']}")
        print(f"Headers: {response['ResponseMetadata']}")

        file_content = response["Body"].read()

        img = Image.open(BytesIO(file_content))
        img_tensor = transform(img).unsqueeze(0)

        output = validation_model(img_tensor)
        _, predicted = torch.max(output, 1)

        is_skin = predicted.item() == 1  # 1 is 'skin', 0 is 'not skin'

        return {"value": is_skin}

    except requests.RequestException as e:
        raise HTTPException(
            status_code=response.status_code, detail=f"Error fetching image: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ClassificationRequestModel(BaseModel):
    url: str = Field(..., description="Parameter to provide url for image scraping.")
    user_id: str = Field(..., description="Parameter to provide a user identifier.")
    timestamp: str = Field(
        ..., description="Parameter to provide a timestamp of request."
    )


@app.post("/classify-skin")
async def classify_skin(request: ClassificationRequestModel):
    url = request.url
    user_id = request.user_id
    timestamp = request.timestamp

    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme != "s3":
            raise ValueError("URL повинен починатися з s3://")

        bucket_name = parsed_url.netloc
        object_key = parsed_url.path.lstrip("/")
        s3 = S3ClientSingleton()

        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        print(f"Status Code: {response['ResponseMetadata']['HTTPStatusCode']}")
        print(f"Headers: {response['ResponseMetadata']}")

        file_content = response["Body"].read()

        img = Image.open(BytesIO(file_content))
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = classification_model(img_tensor)
            probabilities = torch.softmax(output, dim=1).squeeze().tolist()

        result = {CLASS_LABELS[i]: prob for i, prob in enumerate(probabilities)}
        base_path, old_folder, file_name = url.rsplit("/", 2)
        new_file_name = f"{user_id}_{DETECTION_TYPE}_{timestamp}.txt"
        new_s3_path = f"{base_path}/{RESULTS_FOLDER}/{new_file_name}"
        s3_key = "/".join(new_s3_path.split("/")[3:])

        s3.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=json.dumps({**result, "image_url": url}),
            ContentType="application/json",
        )
        return {**result, "path": new_s3_path}

    except requests.RequestException as e:
        raise HTTPException(
            status_code=response.status_code, detail=f"Error fetching image: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RetrainingRequestModel(BaseModel):
    urls: list[str] = Field(..., description="List of S3 URLs for images to retrain on")
    user_id: str = Field(..., description="Parameter to provide a user identifier.")
    timestamp: str = Field(..., description="Parameter to provide a timestamp of request.")

@app.post("/retrain-model")
async def retrain_model(request: RetrainingRequestModel):
    urls = request.urls
    user_id = request.user_id
    timestamp = request.timestamp
    
    # Initialize S3 client
    s3 = S3ClientSingleton()
    
    # Lists to store training data
    training_images = []
    training_labels = []
    
    # Process each image URL
    for url in urls:
        try:
            # Parse S3 URL
            parsed_url = urlparse(url)
            if parsed_url.scheme != "s3":
                continue  # Skip invalid URLs
                
            bucket_name = parsed_url.netloc
            object_key = parsed_url.path.lstrip("/")
            
            # Get image from S3
            response = s3.get_object(Bucket=bucket_name, Key=object_key)
            file_content = response["Body"].read()
            img = Image.open(BytesIO(file_content))
            
            # Find corresponding result file
            base_path, old_folder, file_name = url.rsplit("/", 2)
            result_path = f"{base_path}/{RESULTS_FOLDER}"
            
            # List objects in results folder to find matching result
            result_objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=f"{result_path.split('s3://')[1]}/")
            
            if 'Contents' not in result_objects:
                continue
                
            # Find result file that contains this image URL
            result_file = None
            for obj in result_objects['Contents']:
                result_key = obj['Key']
                result_response = s3.get_object(Bucket=bucket_name, Key=result_key)
                result_content = json.loads(result_response["Body"].read().decode('utf-8'))
                
                if result_content.get('image_url') == url:
                    result_file = result_content
                    break
            
            if not result_file:
                continue
                
            # Extract the class with highest probability
            max_class = max(result_file.items(), key=lambda x: x[1] if x[0] != 'image_url' else -1)
            class_name, probability = max_class
            
            # Only include for retraining if probability is between 0.5 and 0.8
            if 0.5 <= probability < 0.8:
                # Convert image to tensor
                img_tensor = transform(img)
                training_images.append(img_tensor)
                
                # Get label index from class name
                label_idx = CLASS_LABELS.index(class_name)
                training_labels.append(label_idx)
        
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            continue
    
    # If no valid training data found, return early
    if len(training_images) == 0:
        return {"message": "No valid images for retraining found"}
    
    # Convert lists to tensors
    train_images = torch.stack(training_images)
    train_labels = torch.tensor(training_labels)
    
    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Get current model metrics before retraining
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data (assuming you have a test_loader defined elsewhere)
    # For this example, we'll use a small portion of the training data as test data
    test_size = max(1, int(len(train_dataset) * 0.2))
    train_size = len(train_dataset) - test_size
    train_subset, test_subset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
    
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=8, shuffle=False)
    train_subset_loader = torch.utils.data.DataLoader(train_subset, batch_size=8, shuffle=True)
    
    # Get metrics before retraining
    classification_model.to(device)
    old_accuracy, old_precision, old_recall, old_f1 = evaluate(classification_model, device, test_loader)
    
    # Setup for training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classification_model.parameters(), lr=0.001)
    
    # Train the model
    train_loss, train_accuracy, train_precision, train_recall, train_f1 = train(
        classification_model, criterion, optimizer, device, train_subset_loader
    )
    
    # Evaluate after retraining
    new_accuracy, new_precision, new_recall, new_f1 = evaluate(classification_model, device, test_loader)
    
    # Compare metrics
    metrics_improved = (new_accuracy > old_accuracy and new_f1 > old_f1)
    
    # Save model and metrics if improved
    if metrics_improved:
        # Save model to S3
        model_buffer = BytesIO()
        torch.save(classification_model.state_dict(), model_buffer)
        model_buffer.seek(0)
        
        model_key = f"models/classification_model_{timestamp}.pth"
        s3.put_object(
            Bucket=bucket_name,
            Key=model_key,
            Body=model_buffer.getvalue(),
            ContentType="application/octet-stream"
        )
        
        # Save metrics
        metrics = {
            "accuracy": float(new_accuracy),
            "precision": float(new_precision),
            "recall": float(new_recall),
            "f1": float(new_f1),
            "timestamp": timestamp,
            "trained_on_images": len(training_images)
        }
        
        metrics_key = f"models/metrics_{timestamp}.json"
        s3.put_object(
            Bucket=bucket_name,
            Key=metrics_key,
            Body=json.dumps(metrics),
            ContentType="application/json"
        )
        
        return {
            "message": "Model retrained successfully with improved metrics",
            "old_metrics": {
                "accuracy": float(old_accuracy),
                "precision": float(old_precision),
                "recall": float(old_recall),
                "f1": float(old_f1)
            },
            "new_metrics": metrics,
            "model_path": f"s3://{bucket_name}/{model_key}"
        }
    else:
        # Log that metrics got worse
        print(f"Metrics got worse after retraining. Old F1: {old_f1}, New F1: {new_f1}")
        return {
            "message": "Model retraining did not improve metrics",
            "old_metrics": {
                "accuracy": float(old_accuracy),
                "precision": float(old_precision),
                "recall": float(old_recall),
                "f1": float(old_f1)
            },
            "new_metrics": {
                "accuracy": float(new_accuracy),
                "precision": float(new_precision),
                "recall": float(new_recall),
                "f1": float(new_f1)
            }
        }




