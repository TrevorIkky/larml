import tensorflow as tf
from urllib.request import urlopen
from flask import Flask, request, jsonify
from inference import ForestClassificationInference

app = Flask(__name__)
app.config['UPLOAD_EXTENSIONS'] = ['.jpeg', '.jpg', '.png']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.form is not None and "url" in request.form:
            url = request.form["url"]
            req = urlopen(url)
            img = tf.io.decode_image(bytearray(req.read()))
            inference = ForestClassificationInference('./models/forest_classification')
            pred = inference.predict(img)
            return jsonify({'prediction': pred})
        else:
            return jsonify({'message' : 'Request missing param [url]'})
    except Exception:
        return jsonify({'message' : 'An execption was raised when handing prediction.'})

if __name__ == '__main__':
    app.run(host="0.0.0.0")
