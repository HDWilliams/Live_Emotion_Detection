from flask import Flask, request, jsonify, render_template, url_for, send_file
from full_prediction import get_full_prediction

import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def basic_predict():
    if request.method == 'POST':
        img, label = get_full_prediction(request.files['img'])
        output = io.BytesIO()
        img.convert('RGBA').save(output, format='PNG')
        output.seek(0, 0)
        return send_file(output, mimetype='image/png', as_attachment=False)

        #return render_template('results.html', data=data)


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)