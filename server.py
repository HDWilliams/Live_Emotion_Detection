from flask import Flask, request, jsonify, render_template
from full_prediction import get_full_prediction
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def test():
    if request.method == 'POST':
        print(request.files['img'])
        label = get_full_prediction(request.files['img'])
        return label

