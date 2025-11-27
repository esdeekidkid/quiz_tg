import os
from flask import Flask, request, jsonify, session
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup

app = Flask(__name__)
app.secret_key = os.urandom(24)

# In-Memory Session Storage
SESSIONS = {}

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ''
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() + '\n'
    return text

# Function to parse HTML
def parse_html(content):
    soup = BeautifulSoup(content, 'html.parser')
    return soup.get_text()

# Sample route to upload PDF
@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join('/tmp', filename))
        text = extract_text_from_pdf(os.path.join('/tmp', filename))
        return jsonify({'text': text}), 200

# Route to process an incoming request
@app.route('/process', methods=['POST'])
def process_quiz():
    user_id = request.json.get('user_id')
    if user_id not in SESSIONS:
        SESSIONS[user_id] = {'quizzes': []}
    # Process data and update session
    session_data = SESSIONS[user_id]
    session_data['quizzes'].append(request.json.get('quiz'))
    return jsonify({'message': 'Quiz processed successfully'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)