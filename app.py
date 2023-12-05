from flask import Flask, render_template, request
import xgboost as xgb
import numpy as np

app = Flask(__name__)

# Load the trained XGBoost model
model_path = 'model_pickle.xgb'  # Update with your actual path
model = xgb.Booster(model_file=model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        date_numeric = int(request.form['date_numeric'])

        # Make prediction
        prediction = model.predict(xgb.DMatrix(np.array([[date_numeric]])))[0]

        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

