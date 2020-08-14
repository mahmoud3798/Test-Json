import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('KNN_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index.html', prediction_text='Activity will be {}'.format(output))

@app.route('/predict-api',methods=['GET'])
def predict_api():
    '''
    For rendering results on HTML GUI
    '''
    lat = request.args.get('lat')
    lng = request.args.get('lng')
    final_features = [np.array([lat,lng])]
    prediction = model.predict(final_features)

    output = prediction[0]

    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
