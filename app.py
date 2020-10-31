import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_svc.pkl', 'rb'))

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
    prediction_prob=model.predict_proba(final_features)

    output = prediction[0]
    prediction_benign=""
    prediction_malignant=""
    confidence=round(np.max(prediction_prob)*100,2)
    if(output==0):
        prediction_benign='Tumour is Benign'
        statement='Nothing To Worry'
    else:
        prediction_malignant="Tumour is Malignant"
        statement='Please Consult Doctor'

    return render_template('predict.html', prediction_benign=prediction_benign,prediction_malignant=prediction_malignant,statement=statement,confidence="{}% Confidence".format(confidence))


if __name__ == "__main__":
    app.run(debug=True)
