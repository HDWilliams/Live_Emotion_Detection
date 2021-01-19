from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test', methods=['POST'])
def test():
    if request.method == 'POST':
        for f in request.files:
            print(f)
    return


@app.route('/predict', methods=['POST'])
def img_predict():
    return None