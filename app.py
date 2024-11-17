from flask import Flask, send_file
import torch
from transformers import VisionEncoderDecoderModel, AutoProcessor

app = Flask(__name__)

# Load the model with error handling
try:
    model = VisionEncoderDecoderModel.from_pretrained("./half_precision_model")
    processor = AutoProcessor.from_pretrained("./local_nougat_processor")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

@app.route('/')
def index():
    if model_loaded:
        return send_file('index.html')
    else:
        return "There was an error loading the model. Please check the logs for more details.", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
