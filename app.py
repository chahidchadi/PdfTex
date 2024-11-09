from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import os
from PIL import Image
import Pdf_Sliser
import Torch_Main
import torch
from transformers import AutoProcessor, VisionEncoderDecoderModel
from pathlib import Path
import PyPDF2
import base64
from io import BytesIO
from function import final_code_generator, replace_latex_notation
from transformers import AutoTokenizer, NougatProcessor
import sys

# Configuration
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Flask app initialization
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model and Processor loading with error handling
def load_model_and_processor():
    try:
        model = VisionEncoderDecoderModel.from_pretrained("./half_precision_model")
        processor = AutoProcessor.from_pretrained("./local_nougat_processor")
        return model, processor
    except Exception as e:
        print(f"Error loading model or processor: {e}", file=sys.stderr)
        return None, None

model, processor = load_model_and_processor()

if model is None or processor is None:
    raise RuntimeError("Failed to load model or processor. Please check model files and paths.")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

@app.route('/')
def index():
    return send_file('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
