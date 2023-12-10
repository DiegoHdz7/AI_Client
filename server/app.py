

from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


app = Flask(__name__)

# Load the trained model and feature columns
model = joblib.load('model_breast_cancer.pkl')
model_columns = joblib.load('model_breast_cancer_columns.pkl')

# Load the trained model and feature columns
model = joblib.load('tumors_model.pkl')
model_columns = joblib.load('tumors_model_columns.pkl')

@app.route('/predict/tumors', methods=['POST'])
def predictTumor():
    try:
        # Get JSON input from the request
        input_data = request.get_json()

        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input_data], columns=model_columns)

         # Ensure that the input_df has the same columns as the model_columns
        input_df = input_df[model_columns]
    
         # Create MinMaxScaler and scale the features
        minmax_scale = MinMaxScaler()
        input_scaled = minmax_scale.fit_transform(input_df)
    
         # Make predictions
        predictions = model.predict(input_scaled)
        user_predicted_label = np.argmax(predictions)
         # Map numerical labels to class names
        class_names = {0: 'LGG', 1: 'GBM'}
        predicted_class = class_names[user_predicted_label]
    
        # Return the predicted class as JSON
        return jsonify({'prediction': predicted_class,
                         'predictions':predictions.tolist()})
              


    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict/breast-cancer', methods=['POST'])
def predict():
    try:
        # Get JSON input from the request
        input_data = request.get_json()

        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input_data], columns=model_columns)

        # Ensure that the input_df has the same columns as the model_columns
        input_df = input_df[model_columns]

        # Create MinMaxScaler and scale the features
        minmax_scale = MinMaxScaler()
        input_scaled = minmax_scale.fit_transform(input_df)

        # Make predictions
        predictions = model.predict(input_scaled)
        user_predicted_label = np.argmax(predictions)
        # Map numerical labels to class names
        class_names = {0: 'benign', 1: 'malign'}
        predicted_class = class_names[user_predicted_label]

       # Return the predicted class as JSON
        return jsonify({'prediction': predicted_class,
                        'predictions':predictions.tolist()})
        

        # Return the predictions as JSON
        #return jsonify(predictions.tolist())
        # Return the predicted label as JSON
        #return jsonify({'user_predicted_label': int(user_predicted_label)})


    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000)

"""---------------------------------------------------------------------------"""
# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler

# app = Flask(__name__)

# # Load the trained model and feature columns
# model = joblib.load('model_breast_cancer.pkl')
# model_columns = joblib.load('model_breast_cancer_columns.pkl')

# # Create a MinMaxScaler object
# minmax_scale = MinMaxScaler()

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get JSON input from the request
#         input_data = request.get_json()

#         # Create a DataFrame from the input data
#         input_df = pd.DataFrame([input_data], columns=model_columns)

#         # Ensure that the input_df has the same columns as the model_columns
#         input_df = input_df[model_columns]

#         # Scale the input features using the MinMaxScaler
#         input_scaled = minmax_scale.fit_transform(input_df)

#         # Make predictions
#         predictions = model.predict(input_scaled)

       

#         # Include the scaled prediction values in the response
#         scaled_predictions = input_scaled.tolist()[0]

#         # Return the original numerical prediction, the corresponding label, and the scaled predictions as JSON
#         return jsonify({
#             # 'original_prediction': predictions[0],
           
#             'scaled_predictions': scaled_predictions
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(port=5000)




