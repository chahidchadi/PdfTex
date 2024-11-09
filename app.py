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
# Load model directly
from transformers import AutoTokenizer

# Use a pipeline as a high-level helper
from transformers import pipeline
def models():
 # Define local paths for saving the model and processor
 local_model_path = Path("./")
 local_processor_path = Path("./local_nougat_processor")
# Download and save the model
 print("Downloading and saving the model...")
 model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-small")
 model.save_pretrained(local_model_path)
# Download and save the processor
 print("Downloading and saving the processor...")
 processor = AutoProcessor.from_pretrained("facebook/nougat-small")
 processor.save_pretrained(local_processor_path)
 print(f"Model saved to: {local_model_path}")
 print(f"Processor saved to: {local_processor_path}")
 print("Download complete.")
 return model , processor
model , processor = models()
app = Flask(__name__)



@app.route('/')
def index():
    return send_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
