from flask import Flask, request, jsonify, send_file
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
from function import final_code_generator, models, replace_latex_notation
import gc  # Added garbage collection for memory management

app = Flask(__name__)
processor = AutoProcessor.from_pretrained("./local_nougat_processor")
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensures folder is created if not exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

conversion_results = {}

# Load model globally to avoid reloading multiple times


def count_pdf_pages(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
        return num_pages
    except (FileNotFoundError, PyPDF2.errors.PdfReadError) as e:
        return f"Error: {str(e)}"

def generate_thumbnails(filepath):
    images = Pdf_Sliser.rasterize_paper(pdf=filepath, return_pil=True)
    thumbnails = []
    for img in images:
        with Image.open(img) as image:  # Use context manager to release memory after processing
            image.thumbnail((100, 100))  # Resize to thumbnail
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            thumbnails.append(img_str)
    # Explicit garbage collection
    del images
    gc.collect()
    return thumbnails

def pdf_to_latex(filepath, page_number):
    images = Pdf_Sliser.rasterize_paper(pdf=filepath, return_pil=True)
    if page_number < 1 or page_number > len(images):
        raise ValueError(f"Invalid page number. The PDF has {len(images)} pages.")

    with Image.open(images[page_number - 1]) as image:  # Use context manager for memory management
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        model = VisionEncoderDecoderModel.from_pretrained("./half_precision_model")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        outputs = model.generate(
            pixel_values.to(device),
            min_length=1,
            max_length=3584,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_scores=True,
            stopping_criteria=Torch_Main.StoppingCriteriaList([Torch_Main.StoppingCriteriaScores()]),
        )
    generated = processor.batch_decode(outputs[0], skip_special_tokens=True)[0]
    generated = processor.post_process_generation(generated, fix_markdown=False)
    
    # Explicit memory cleanup
    del pixel_values, outputs, images
    gc.collect()
    
    return generated

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file uploaded'}), 400

    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            num_pages = count_pdf_pages(filepath)
            thumbnails = generate_thumbnails(filepath)
            return jsonify({
                'message': 'PDF uploaded successfully',
                'filename': filename,
                'num_pages': num_pages,
                'thumbnails': thumbnails
            })
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up uploaded files to avoid accumulation
            if os.path.exists(filepath):
                os.remove(filepath)
    else:
        return jsonify({'error': 'Invalid file type. Please upload a PDF.'}), 400

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    filename = request.json.get('filename')
    page_number = request.json.get('page_number')

    if not filename or not page_number:
        return jsonify({'error': 'Missing filename or page number'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    try:
        latex_output = pdf_to_latex(filepath, page_number)
        latex_output = final_code_generator(latex_output)
        latex_output = rf"{latex_output}"
        latex_output = replace_latex_notation(latex_output)

        # Store results in memory
        conversion_results[filename] = latex_output
        return jsonify({'message': 'Conversion complete', 'filename': filename})
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up to free memory
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/get_results/<filename>', methods=['GET'])
def get_results(filename):
    if filename in conversion_results:
        return jsonify(conversion_results[filename])
    else:
        return jsonify({'error': 'Results not found'}), 404

@app.route('/about.html')
def about():
    return send_file('about.html')

@app.route('/contact.html')
def contact():
    return send_file('contact.html')

@app.route('/donate.html')
def donate():
    return send_file('donate.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
