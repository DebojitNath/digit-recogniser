# Digit Recognizer (Active Learning AI)

A web-based application for recognizing handwritten digits using a Convolutional Neural Network (CNN). This project features a **PyTorch backend** running on **Django**, a modern Glassmorphism UI, and an **Active Learning (Human-in-the-Loop)** feedback system.

## 🌐 Live Demo

Try the live application here: **[https://digit-recogniser-h5a2.onrender.com/](https://digit-recogniser-h5a2.onrender.com/)**

## Features

- **Interactive Canvas**: Draw digits smoothly with a thick brush designed to match the MNIST dataset format.
- **Real-time Inference**: Sends the drawn image to a PyTorch backend for high-speed prediction.
- **Dynamic Probabilities**: Animated progress bars showing the model's confidence for all digits (0-9).
- **Active Online Learning (Human-in-the-Loop)**: If the model guesses wrong, provide the correct answer! The model instantly runs backpropagation on your single image and updates its internal weights (`mnist_cnn.pth`) to learn from its mistake.
- **Modern UI**: A responsive, dark-mode Glassmorphism design with floating animated background blobs.

## Technologies Used

- **Backend**: Django 1.10, Python
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism), JavaScript
- **Machine Learning**: PyTorch (`torch`, `torchvision`), PIL (Pillow)
- **Model**: Custom Sequential CNN trained on the MNIST dataset

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DebojitNath/digit-recogniser.git
   cd digit-recogniser
   ```

2. **Install Python dependencies:**
   Make sure you have Python installed, then install Django and PyTorch:
   ```bash
   pip install django==1.10
   pip install torch torchvision pillow
   ```

3. **Train the Initial Model:**
   Before running the server, you need to generate the initial weights file by running the PyTorch training script:
   ```bash
   python train_pytorch.py
   ```
   *(This takes about 1-2 minutes and will generate a `mnist_cnn.pth` file in the root directory).*

4. **Run the Django Server:**
   ```bash
   python manage.py runserver
   ```

5. **Open your browser:**
   - **Local:** Navigate to `http://127.0.0.1:8000/`
   - **Live Production Deployment:** [https://digit-recogniser-h5a2.onrender.com/](https://digit-recogniser-h5a2.onrender.com/)

## How Active Learning Works

1. Draw a digit on the canvas and click "Predict".
2. The UI will ask "Was this correct? 👍 👎".
3. If you click 👎, an input field will appear asking for the correct digit.
4. When you click **"Teach Model"**, the backend switches the PyTorch model to `train()` mode, calculates the loss against your provided correct digit, runs a single step of gradient descent (`optimizer.step()`), and overwrites `mnist_cnn.pth`.
5. The model instantly gets smarter based on your direct feedback!

## Project Structure

```
digit-recogniser/
├── api/                    # Django app
│   ├── static/api/         # CSS and JS files
│   ├── templates/api/      # HTML templates (Glassmorphism UI)
│   ├── views.py            # PyTorch Inference & Online Learning logic
│   └── urls.py             # URL routing for predict and feedback endpoints
├── dlserver/               # Django project settings
├── train_pytorch.py        # PyTorch training script (MNIST Dataset)
├── manage.py               # Django management script
└── README.md
```

## Acknowledgments
- MNIST dataset by Yann LeCun and Corinna Cortes
- PyTorch team for the amazing deep learning framework
- Django community for the robust web framework
