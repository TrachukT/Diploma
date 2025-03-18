import json

import boto3
from io import BytesIO
from PIL import Image
import html
import re

# Функція для отримання файлу з S3
def get_s3_file(bucket_name, object_key, transform):
    # Ініціалізація клієнта S3
    s3 = boto3.client('s3')

    # Завантаження об'єкта в пам'ять
    try:
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        print(f"Status Code: {response['ResponseMetadata']['HTTPStatusCode']}")
        print(f"Headers: {response['ResponseMetadata']}")

        # Отримання вмісту файлу як потоку байтів
        file_content = response['Body'].read()

        # Завантаження файлу як зображення
        img = Image.open(BytesIO(file_content))
        img_tensor = transform(img).unsqueeze(0)

        return img_tensor
    except Exception as e:
        print(f"Error fetching the file: {e}")
        return None


# Приклад використання
if __name__ == "__main__":
    # S3 інформація
    bucket_name = "diplomatest"
    object_key = "0_0.jpg"
    s3 = boto3.client('s3')
    new_file_name = f"user_id_DETECTION_TYPE_timestamp.txt"
    new_s3_path = f"s3://diplomatest/user_data/uiid/29.01.2025/results/{new_file_name}"
    s3_key = "/".join(new_s3_path.split("/")[3:])
    result ={
        "value": True
    }
    s3.put_object(Bucket=bucket_name, Key=s3_key, Body=json.dumps({**result, "image_url": "dldklfdlfk"}), ContentType="application/json")
    print({**result, "path": new_s3_path})
