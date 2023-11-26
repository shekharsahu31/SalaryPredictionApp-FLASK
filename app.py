import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    prediction = model.predict(final_features)

    exp = float(request.form['experience'])
    ts = float(request.form['test_score'])
    its = float(request.form['interview_score'])
    array = [exp , ts , its]
    final_features2 = [np.array(array)]

    prediction2 = model.predict(final_features2)

    output = round(prediction[0], 2)
    output2 = round(prediction2[0], 2)

    return render_template('index.html', prediction_text=f'Employee Salary should be $ {output} and {output2}')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)