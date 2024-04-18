from flask import Flask, request, jsonify
import pickle
import pandas as pd

filename = 'models/heart-disease-prediction-model.pkl'
with open(filename, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)


@app.route('/')
def home():
    return "Heart Disease Prediction API"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get data posted as JSON

        input_df = pd.DataFrame([[
            data['age'], data['sex'], data['cp'], data['trestbps'],
            data['chol'], data['fbs'], data['restecg'], data['thalach'],
            data['exang'], data['oldpeak'], data['slope']
        ]], columns=['Age', 'Sex', 'ChestPainType', 'RestingBP',
                     'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR',
                     'ExerciseAngina', 'Oldpeak', 'ST_Slope'])

        # Predict probabilities using the DataFrame to maintain feature names
        probabilities = model.predict_proba(input_df)
        # Probability of the positive class (e.g., index 1 for binary classifiers)
        positive_class_probability = probabilities[0][1]

        # Convert probability to percentage
        print(positive_class_probability)
        prediction_percentage = round(positive_class_probability * 100, 2)

        # Send back the prediction result as a percentage
        return jsonify({'prediction_percentage': prediction_percentage})

    except Exception as e:
        # Return the error message if an exception occurs
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
