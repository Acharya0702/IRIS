import flask
from flask import Flask
import sklearn.externals
import joblib
from flask import render_template, request
import numpy as np
app = Flask(__name__)
model = joblib.load('model.pkl')
@app.route("/")
def home():
   return render_template('index.html')
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':
        f1 = request.form['f1']
        f2 = request.form['f2']
        f3 = request.form['f3']
        f4 = request.form['f4']
        feature_array = [f1,f2,f3,f4]
        feature = np.asarray(feature_array, dtype='float64').reshape(1,-1)
        prediction = model.predict(feature)
        if prediction == 0:
            prediction='Iris-setosa'
        elif prediction == 1:
            prediction='Iris-versicolor'
        else:
            prediction='Iris-virginica'
        return render_template('index.html', prediction='{}'.format(prediction))
if __name__ == '__main__':
    app.run(debug=True)