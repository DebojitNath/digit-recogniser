import base64
import os
import io
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

# 1. We must define the model architecture here so PyTorch knows how to load the weights
class DigitalCNN(nn.Module):
    def __init__(self):
        super(DigitalCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(4608, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# 2. Load the trained model into memory (we do this outside the view so it only loads once when the server starts)
model = DigitalCNN()
# Load the trained model weights
MODEL_PATH = os.environ.get('MODEL_PATH', 'mnist_cnn.pth')
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    print(f"Failed to load model weights from {MODEL_PATH}: {e}") # Set the model to evaluation (testing) mode

# 3. Setup the image processor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ================== DJANGO VIEWS ==================

def index(request):
    print("get req")
    return render(request, 'api/main.html')

@csrf_exempt
def predict_digit(request):
    if request.method == 'POST':
        # Get the image data sent from JavaScript
        data = json.loads(request.body)
        img_base64 = data.get('image').split(',')[1] # Strip the "data:image/png;base64," header
        
        # Convert the base64 string into a real Image
        image = Image.open(io.BytesIO(base64.b64decode(img_base64)))
        
        # Preprocess the image and add a batch dimension
        tensor = transform(image).unsqueeze(0)
        
        # Run inference using PyTorch!
        with torch.no_grad():
            output = model(tensor)
            
            # Get the exact probabilities for each digit (0-9)
            probabilities = output[0].tolist()
            # Get the index of the highest probability
            predicted_class = int(torch.argmax(output, dim=1).item())

        # Send it back to the browser
        return JsonResponse({
            'prediction': predicted_class,
            'probabilities': [round(p * 100, 2) for p in probabilities]
        })

@csrf_exempt
def submit_feedback(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            img_base64 = data.get('image').split(',')[1]
            correct_label = int(data.get('correct_label'))
            
            image = Image.open(io.BytesIO(base64.b64decode(img_base64)))
            tensor = transform(image).unsqueeze(0)
            
            # Switch to training mode
            model.train()
            
            # Define loss and optimizer with a small learning rate for fine-tuning
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
            
            # Single step of backpropagation
            optimizer.zero_grad()
            output = model(tensor)
            
            # The target must be a 1D tensor containing the class index
            target = torch.tensor([correct_label])
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Save the updated weights so it remembers!
            torch.save(model.state_dict(), MODEL_PATH)
            
            # Switch back to eval mode for future predictions
            model.eval()
            
            return JsonResponse({'status': 'success', 'loss': float(loss.item())})
        except Exception as e:
            print(f"Error in feedback: {e}")
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'status': 'invalid method'}, status=405)
