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
from PIL import Image
import io
from io import BytesIO
from function import final_code_generator
from function import models , replace_latex_notation


app = Flask(__name__)
model = VisionEncoderDecoderModel.from_pretrained("./local_nougat_model")
processor = AutoProcessor.from_pretrained("./local_nougat_processor")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


conversion_results = {}

@app.route('/')
def index():
    return send_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
