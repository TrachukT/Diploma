import json
from urllib.parse import urlparse

import boto3
from fastapi import HTTPException, APIRouter
from app.schemas.models import *
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
import torch.nn as nn
from io import BytesIO
from models.training.NN_class import ConvNeuralNet  # Import your model class
from models.training.utils import train, evaluate
from config import CLASS_LABELS, DETECTION_TYPE, RESULTS_FOLDER
from typing import Optional

router = APIRouter()


class S3ClientSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = boto3.client("s3")
        return cls._instance


transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

validation_model = ConvNeuralNet(num_classes=2)
validation_model.load_state_dict(torch.load("C:/Diploma/Diploma/app/internal/models/files/val_model.pth"))
validation_model.eval()

classification_model = ConvNeuralNet(num_classes=7)
classification_model.load_state_dict(
    torch.load("C:/Diploma/Diploma/app/internal/models/files/custom_classification_model.pth")
)
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


@router.post("/retrain-model", response_model=RetrainingResponse)
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
            result_objects = s3.list_objects_v2(
                Bucket=bucket_name, Prefix=f"{result_path.split('s3://')[1]}/"
            )

            if "Contents" not in result_objects:
                continue

            # Find result file that contains this image URL
            result_file = None
            for obj in result_objects["Contents"]:
                result_key = obj["Key"]
                result_response = s3.get_object(Bucket=bucket_name, Key=result_key)
                result_content = json.loads(
                    result_response["Body"].read().decode("utf-8")
                )

                if result_content.get("image_url") == url:
                    result_file = result_content
                    break

            if not result_file:
                continue

            # Extract the class with highest probability
            # Видаляємо 'image_url' зі словника перед пошуком максимуму
            classification_results = {
                k: v for k, v in result_file.items() if k != "image_url"
            }
            max_class = max(classification_results.items(), key=lambda x: x[1])
            class_name, probability = max_class

            # Додаємо зображення для тренування тільки якщо ймовірність < 0.8
            if probability < 0.8:
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
        return RetrainingResponse(
            message="No valid images for retraining found",
            old_metrics=MetricsResponse(accuracy=0, precision=0, recall=0, f1=0),
            new_metrics=MetricsResponse(accuracy=0, precision=0, recall=0, f1=0),
        )

    # Convert lists to tensors
    train_images = torch.stack(training_images)
    train_labels = torch.tensor(training_labels)

    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True
    )

    # Get current model metrics before retraining
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Знаходимо останню версію моделі
    try:
        model_objects = s3.list_objects_v2(
            Bucket=bucket_name, Prefix="models/classification_model_"
        )

        if "Contents" in model_objects and model_objects["Contents"]:
            # Сортуємо за датою і беремо найновішу
            latest_model = sorted(
                model_objects["Contents"], key=lambda x: x["LastModified"], reverse=True
            )[0]
            latest_model_key = latest_model["Key"]

            # Завантажуємо останню версію моделі
            latest_model_response = s3.get_object(
                Bucket=bucket_name, Key=latest_model_key
            )
            latest_model_buffer = BytesIO(latest_model_response["Body"].read())

            # Інціалізуємо і завантажуємо останню версію моделі
            latest_classification_model = ConvNeuralNet(num_classes=7)
            latest_classification_model.load_state_dict(torch.load(latest_model_buffer))
            latest_classification_model.to(device)
            latest_classification_model.eval()
        else:
            # Якщо немає попередніх моделей, використовуємо поточну
            latest_classification_model = classification_model
    except Exception as e:
        print(f"Error loading latest model: {str(e)}")
        latest_classification_model = classification_model

    # Розділення на train і test
    test_size = max(1, int(len(train_dataset) * 0.2))
    train_size = len(train_dataset) - test_size
    train_subset, test_subset = torch.utils.data.random_split(
        train_dataset, [train_size, test_size]
    )

    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=8, shuffle=False)
    train_subset_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=8, shuffle=True
    )

    # Отримуємо метрики перед ретренуванням
    old_accuracy, old_precision, old_recall, old_f1 = evaluate(
        latest_classification_model, device, test_loader
    )

    # Клонуємо модель для ретренування
    model_to_train = ConvNeuralNet(num_classes=7)
    model_to_train.load_state_dict(latest_classification_model.state_dict())
    model_to_train.to(device)

    # Налаштування для тренування
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_to_train.parameters(), lr=0.001)

    # Тренуємо модель
    train_loss, train_accuracy, train_precision, train_recall, train_f1 = train(
        model_to_train, criterion, optimizer, device, train_subset_loader
    )

    # Оцінюємо після ретренування
    new_accuracy, new_precision, new_recall, new_f1 = evaluate(
        model_to_train, device, test_loader
    )

    # Порівнюємо метрики
    metrics_improved = new_accuracy > old_accuracy and new_f1 > old_f1

    # Зберігаємо модель і метрики, якщо покращились
    if metrics_improved:
        # Зберігаємо модель в S3
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

        # Зберігаємо метрики
        metrics = {
            "accuracy": float(new_accuracy),
            "precision": float(new_precision),
            "recall": float(new_recall),
            "f1": float(new_f1),
            "timestamp": timestamp,
            "trained_on_images": len(training_images),
        }

        metrics_key = f"models/metrics_{timestamp}.json"
        s3.put_object(
            Bucket=bucket_name,
            Key=metrics_key,
            Body=json.dumps(metrics),
            ContentType="application/json",
        )

        return RetrainingResponse(
            message="Model retrained successfully with improved metrics",
            old_metrics=MetricsResponse(
                accuracy=float(old_accuracy),
                precision=float(old_precision),
                recall=float(old_recall),
                f1=float(old_f1),
            ),
            new_metrics=MetricsResponse(
                accuracy=float(new_accuracy),
                precision=float(new_precision),
                recall=float(new_recall),
                f1=float(new_f1),
            ),
            model_path=f"s3://{bucket_name}/{model_key}",
        )
    else:
        # Логуємо, що метрики погіршились
        print(f"Metrics got worse after retraining. Old F1: {old_f1}, New F1: {new_f1}")
        return RetrainingResponse(
            message="Model retraining did not improve metrics",
            old_metrics=MetricsResponse(
                accuracy=float(old_accuracy),
                precision=float(old_precision),
                recall=float(old_recall),
                f1=float(old_f1),
            ),
            new_metrics=MetricsResponse(
                accuracy=float(new_accuracy),
                precision=float(new_precision),
                recall=float(new_recall),
                f1=float(new_f1),
            ),
        )
