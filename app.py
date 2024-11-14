from flask import Flask, send_file
import os

app = Flask(__name__)

@app.route('/')
def index():
    return send_file('index.html')

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
