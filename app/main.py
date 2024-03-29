from flask import request, Flask, render_template
from flask_cors import CORS
import json
from ml_func import makePrediction
from werkzeug.utils import secure_filename
from helpers import file_manager
import os

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = file_manager.UPLOAD_FOLDER

# enable cors
CORS(app)


@app.route("/")
def home_view():
    return render_template('index.html')


@app.route('/test', methods=['POST'])
def test():
    return {
        "test": "test is working"
    }

# Example endpoint


@app.route('/ml/predict', methods=['POST'])
def predict():
    current_filename = ''
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return {
                "error": "No file part"
            }
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return {
                "error": "No selected file"
            }
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
