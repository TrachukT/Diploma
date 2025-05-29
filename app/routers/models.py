import io
import json
import os
from datetime import datetime
from urllib.parse import urlparse

import boto3
from dotenv import load_dotenv
from fastapi import HTTPException, APIRouter, Depends
from sqlalchemy.orm import Session
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models

from app.db.config import get_db
from app.db.model_metrics import ModelMetrics
from app.schemas.models import *
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
import torch.nn as nn
from io import BytesIO
from app.internal.models.training.NN_class import (
    ConvNeuralNet,
)  # Import your model class
from app.internal.models.training.utils import train, evaluate
from app.config import CLASS_LABELS, DETECTION_TYPE, RESULTS_FOLDER

router = APIRouter()

NUM_CLASSES = 7


class S3ClientSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = boto3.client("s3")
        return cls._instance


valid_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)
classif_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

load_dotenv()
s3 = S3ClientSingleton()
bucket_name = os.getenv("S3_BUCKET_NAME")


def load_model_from_s3(bucket, key, model, device):
    response = s3.get_object(Bucket=bucket, Key=key)
    buffer = io.BytesIO(response["Body"].read())
    state_dict = torch.load(buffer, map_location=device)
    model.load_state_dict(state_dict)
    return model


device = torch.device("cpu")

validation_model = ConvNeuralNet(num_classes=2)
validation_model = load_model_from_s3(
    bucket_name, "models/validation_model.pth", validation_model, device
)
validation_model.eval()

classification_model = models.mobilenet_v2(pretrained=False)
classification_model.classifier = nn.Sequential(
    nn.Dropout(0.3), nn.Linear(classification_model.last_channel, NUM_CLASSES)
)
classification_model_key = "models/mobilenet_skin_disease_model.pth"
classification_model = load_model_from_s3(
    bucket_name, classification_model_key, classification_model, device
)
classification_model.to(device)
classification_model.eval()


@router.post("/validate-skin")
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
        img_tensor = valid_transform(img).unsqueeze(0)

        output = validation_model(img_tensor)
        _, predicted = torch.max(output, 1)

        is_skin = predicted.item() == 1  # 1 is 'skin', 0 is 'not skin'

        return {"is_skin": is_skin}

    except requests.RequestException as e:
        raise HTTPException(
            status_code=response.status_code, detail=f"Error fetching image: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify-skin")
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
        img_tensor = classif_transform(img).unsqueeze(0)

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
        return {**result, "results_path": new_s3_path, "image_url": url}

    except requests.RequestException as e:
        raise HTTPException(
            status_code=response.status_code, detail=f"Error fetching image: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrain-model", response_model=RetrainingResponse)
async def retrain_model(request: RetrainingRequestModel, db: Session = Depends(get_db)):
    urls = request.urls
    timestamp = request.timestamp

    s3 = S3ClientSingleton()

    training_images = []
    training_labels = []
    bucket_name = None
    for url in urls:
        try:
            parsed_url = urlparse(url)
            if parsed_url.scheme != "s3":
                continue

            bucket_name = parsed_url.netloc
            object_key = parsed_url.path.lstrip("/")

            response = s3.get_object(Bucket=bucket_name, Key=object_key)
            response_content = response["Body"].read()
            result_response = json.loads(response_content.decode("utf-8"))

            classification_results = {
                k: v for k, v in result_response.items() if k != "image_url"
            }
            max_class = max(classification_results.items(), key=lambda x: x[1])
            class_name, probability = max_class

            if probability < 0.8:
                parsed_url = urlparse(result_response.get("image_url"))
                bucket_name = parsed_url.netloc
                object_key = parsed_url.path.lstrip("/")

                response = s3.get_object(Bucket=bucket_name, Key=object_key)
                file_content = response["Body"].read()

                img = Image.open(BytesIO(file_content)).convert("RGB")
                training_images.append(img)

                label_idx = next(k for k, v in CLASS_LABELS.items() if v == class_name)
                training_labels.append(label_idx)

        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            continue

    if len(training_images) == 0:
        return RetrainingResponse(
            message="No valid images for retraining found",
            old_metrics=MetricsResponse(accuracy=0, precision=0, recall=0, f1=0),
            new_metrics=MetricsResponse(accuracy=0, precision=0, recall=0, f1=0),
        )

    dataset_size = len(training_images)
    test_size = max(1, int(dataset_size * 0.4))
    train_size = dataset_size - test_size

    train_imgs = training_images[:train_size]
    train_lbls = training_labels[:train_size]
    test_imgs = training_images[train_size:]
    test_lbls = training_labels[train_size:]

    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    train_tensors = [train_transforms(img) for img in train_imgs]
    test_tensors = [test_transforms(img) for img in test_imgs]

    train_dataset = TensorDataset(torch.stack(train_tensors), torch.tensor(train_lbls))
    test_dataset = TensorDataset(torch.stack(test_tensors), torch.tensor(test_lbls))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    latest_metrics = (
        db.query(ModelMetrics)
        .filter(ModelMetrics.model_type == DETECTION_TYPE)
        .order_by(ModelMetrics.created_at.desc())
        .first()
    )

    model_to_train = models.mobilenet_v2(pretrained=False)
    model_to_train.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(model_to_train.last_channel, NUM_CLASSES)
    )
    model_to_train.load_state_dict(classification_model.state_dict())
    model_to_train.to(device)

    # Налаштування для тренування
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_to_train.parameters(), lr=0.001)
    num_epochs = 10
    train_loss, train_accuracy, train_precision, train_recall, train_f1 = 0, 0, 0, 0, 0
    eval_accuracy, eval_precision, eval_recall, eval_f1 = 0, 0, 0, 0
    for epoch in range(num_epochs):
        train_loss, train_accuracy, train_precision, train_recall, train_f1 = train(
            model_to_train, criterion, optimizer, device, train_loader
        )

        eval_accuracy, eval_precision, eval_recall, eval_f1 = evaluate(
            model_to_train, device, test_loader
        )
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(
            f"Train - Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}, Prec: {train_precision:.4f}, Rec: {train_recall:.4f},F1: {train_f1:.4f}"
        )
        print(
            f"Eval  - Acc: {eval_accuracy:.4f}, Prec: {eval_precision:.4f}, Rec: {eval_recall:.4f}, F1: {eval_f1:.4f}"
        )

    old_train_acc = latest_metrics.training_accuracy or 0
    old_train_f1 = latest_metrics.training_f1_score or 0
    old_eval_acc = latest_metrics.evaluation_accuracy or 0
    old_eval_f1 = latest_metrics.evaluation_f1_score or 0

    train_improved = train_accuracy > old_train_acc and train_f1 > old_train_f1
    eval_improved = eval_accuracy > old_eval_acc and eval_f1 > old_eval_f1

    f1_gap = abs(train_f1 - eval_f1)
    max_allowed_gap = 0.18

    not_overfitting = f1_gap < max_allowed_gap

    if train_improved and eval_improved and not_overfitting:
        model_buffer = BytesIO()
        torch.save(model_to_train.state_dict(), model_buffer)
        model_buffer.seek(0)

        model_key = f"models/classification_model_{timestamp}.pth"
        s3.put_object(
            Bucket=bucket_name,
            Key=model_key,
            Body=model_buffer.getvalue(),
            ContentType="application/octet-stream",
        )

        s3.put_object(
            Bucket=bucket_name,
            Key=classification_model_key,
            Body=model_buffer.getvalue(),
            ContentType="application/octet-stream",
        )

        metrics = ModelMetrics(
            model_type=DETECTION_TYPE,
            training_loss=train_loss,
            training_accuracy=train_accuracy,
            training_precision=train_precision,
            training_recall=train_recall,
            training_f1_score=train_f1,
            evaluation_accuracy=eval_accuracy,
            evaluation_precision=eval_precision,
            evaluation_recall=eval_recall,
            evaluation_f1_score=eval_f1,
            created_at=datetime.utcnow(),
        )
        db.add(metrics)
        db.commit()

        return RetrainingResponse(
            message="Model retrained successfully with improved metrics and no overfitting",
            old_metrics=MetricsResponse(
                accuracy=float(old_eval_acc),
                precision=float(latest_metrics.evaluation_precision or 0),
                recall=float(latest_metrics.evaluation_recall or 0),
                f1=float(old_eval_f1),
            ),
            new_metrics=MetricsResponse(
                accuracy=float(eval_accuracy),
                precision=float(eval_precision),
                recall=float(eval_recall),
                f1=float(eval_f1),
            ),
            model_path=f"s3://{bucket_name}/{model_key}",
        )
    else:
        print(
            f"Train improved: {train_improved}, Eval improved: {eval_improved}, F1 gap: {f1_gap}"
        )
        return RetrainingResponse(
            message="Model retraining did not improve metrics or overfitting detected",
            old_metrics=MetricsResponse(
                accuracy=float(old_eval_acc),
                precision=float(latest_metrics.evaluation_precision or 0),
                recall=float(latest_metrics.evaluation_recall or 0),
                f1=float(old_eval_f1),
            ),
            new_metrics=MetricsResponse(
                accuracy=float(eval_accuracy),
                precision=float(eval_precision),
                recall=float(eval_recall),
                f1=float(eval_f1),
            ),
        )
