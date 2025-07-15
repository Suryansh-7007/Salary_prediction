import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    experience = float(request.form['experience'])
    test_score = float(request.form['test_score'])
    interview_score = float(request.form['interview_score'])

    usd_salary = model.predict([[experience, test_score, interview_score]])[0]
    conversion_rate = 83
    inr_salary = usd_salary * conversion_rate

    return render_template('index.html', prediction_text='Estimated Salary in ₹: {:.2f}'.format(inr_salary))

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
    from os import getenv
    port = int(getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
