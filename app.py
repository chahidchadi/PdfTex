from flask import Flask, jsonify
from transformers import VisionEncoderDecoderModel, AutoProcessor

app = Flask(__name__)

@app.route('/')
def load_model():
    try:
        model = VisionEncoderDecoderModel.from_pretrained("./half_precision_model")
        return jsonify({"message": "The model is loaded successfully."})
    except Exception as e:
        return jsonify({"message": f"The model is not loaded. Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
