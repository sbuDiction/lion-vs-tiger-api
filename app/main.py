from flask import Flask
from flask import request
from flask_cors import CORS
import sqlite3
import json
from ml_func import makePrediction
from werkzeug.utils import secure_filename
from helpers import file_manager
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = file_manager.UPLOAD_FOLDER

# enable cors
CORS(app)


@app.route("/")
def home_view():
    return {
        "api_status": "Up and running"
    }


# Example endpoint
@app.route('/ml/predict', methods=['POST'])
def predict():
    current_filename = ''
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            print('No selected file')
        if file and file_manager.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            current_filename = filename

    predicted_results = makePrediction(current_filename)
    print(predicted_results)
    # response
    return {
        "status": "success",
        "prediction": predicted_results
    }
