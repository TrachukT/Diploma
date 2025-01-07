from flask import Flask, jsonify, request
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
from NN_class import ConvNeuralNet  # Import your model class

app = Flask(__name__)

transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),         
    transforms.Normalize(mean=[0.5], std=[0.5])  
])

# Load the model
model = ConvNeuralNet(num_classes=2)
model.load_state_dict(torch.load("model.pth"))
model.eval()

@app.route('/validate-skin', methods=['POST'])
def validate_skin():
    data = request.json
    if 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400
    
    url = data['url']
    print(url)
    
    try:
        # Fetch image with a timeout
        response = requests.get(url, timeout=10)  # Timeout after 10 seconds
        response.raise_for_status()
        
        # Logging the status and headers for debugging
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {response.headers}")
        
        img = Image.open(BytesIO(response.content))
        
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        
        is_skin = predicted.item() == 1  # Assuming 1 is 'skin' and 0 is 'not skin'
        
        return jsonify({"value": is_skin})
    
    except requests.RequestException as e:
        return jsonify({"error": f"Error fetching image: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
