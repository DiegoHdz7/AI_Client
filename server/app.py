

from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model and feature columns
model_breast_cancer = joblib.load('model_breast_cancer.pkl')
model_columns__breast_cancer = joblib.load('model_breast_cancer_columns.pkl')

# Load the trained model and feature columns
model_tumors = joblib.load('tumors_model.pkl')
model_columns_tumors = joblib.load('tumors_model_columns.pkl')

@app.route('/predict/tumors', methods=['POST'])
def predictTumor():
    try:
        
        # Get JSON input from the request
        input_data = request.get_json()

        # Print the received input data for debugging
        print('Received Data:')
        print(input_data)

        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input_data], columns=model_columns_tumors)

        # Ensure that the input_df has the same columns as the model_columns
        input_df = input_df[model_columns_tumors]

        # Print the processed input data for debugging
        print('Processed Input Data:')
        print(input_df)

        # Create MinMaxScaler and scale the features
        minmax_scale = MinMaxScaler()
        input_scaled = minmax_scale.fit_transform(input_df)

        # Print the scaled input data for debugging
        print('Scaled Input:')
        print(input_scaled)

        # Make predictions
        predictions = model_tumors.predict(input_scaled)
        user_predicted_label = np.argmax(predictions)

        # Map numerical labels to class names
        class_names = {0: 'LGG', 1: 'GBM'}
        predicted_class = class_names[user_predicted_label]

        # Print predictions for debugging
        print('Prediction Label:')
        print(predicted_class)
        print('Predictions:')
        print(predictions)

        # Return the predicted class as JSON
        return {'prediction': predicted_class,
                'predictions': predictions.tolist()}
              


    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict/breast-cancer', methods=['POST'])
def predict():
    try:
        # Get JSON input from the request
        input_data = request.get_json()

        # Print the received input data for debugging
        print('Received Data:')
        print(input_data)

        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input_data], columns=model_columns__breast_cancer)

        # Ensure that the input_df has the same columns as the model_columns
        input_df = input_df[model_columns__breast_cancer]

        # Print the processed input data for debugging
        print('Processed Input Data:')
        print(input_df)

        # Create MinMaxScaler and scale the features
        minmax_scale = MinMaxScaler()
        input_scaled = minmax_scale.fit_transform(input_df)

        # Print the scaled input data for debugging
        print('Scaled Input:')
        print(input_scaled)

        # Make predictions
        predictions = model_breast_cancer.predict(input_scaled)
        user_predicted_label = np.argmax(predictions)

        # Map numerical labels to class names
        class_names = {0: 'benign', 1: 'malign'}
        predicted_class = class_names[user_predicted_label]

        # Print predictions for debugging
        print('Prediction Label:')
        print(predicted_class)
        print('Predictions:')
        print(predictions)

        # Return the predicted class as JSON
        return {'prediction': predicted_class,
                'predictions': predictions.tolist()}

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000)