from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your model during the server initialization
Ann_model = load_model(r'C:\Users\ashis\Downloads\Churn-Analysis-using-Ann--Artificial-Neural-Network-\Ann_model')  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        features = [
            float(request.form['Credit score']),
            float(request.form['Country']),
            float(request.form['gender']),
            float(request.form['Age']),
            float(request.form['Tenure']),
            float(request.form['Balance']),
            float(request.form['Products']),
            float(request.form['Credit card']),
            float(request.form['Active member']),
            float(request.form['Estimated Salary'])
        ]

        # Convert features to a numpy array
        features_np = np.array(features).reshape(1, -1)

        # Make prediction using the loaded model
        prediction = Ann_model.predict(features_np)

        # Assuming a binary classification, you can threshold the prediction
        binary_prediction = 1 if prediction[0, 0] > 0.3 else 0

        # Pass the prediction to the result.html template
        return render_template('index.html', prediction=int(binary_prediction))

    except Exception as e:
        # Pass the error to the result.html template
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(port=5000)
