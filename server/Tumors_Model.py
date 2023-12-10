from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# fetch dataset 
glioma_grading_clinical_and_mutation_features = fetch_ucirepo(id=759) 

# Convert data to Pandas DataFrame
X = pd.DataFrame(glioma_grading_clinical_and_mutation_features.data.features)
y = pd.DataFrame(glioma_grading_clinical_and_mutation_features.data.targets)

# metadata 
print(glioma_grading_clinical_and_mutation_features.metadata) 
  
# variable information 
print(glioma_grading_clinical_and_mutation_features.variables)

# Assuming df is your DataFrame
X['Race'] = X['Race'].map({'white': 0, 'black or african american': 1, 'asian': 2})

# If there are NaN values in the 'Race' column, you can fill them with a default value, e.g., -1
X['Race'] = X['Race'].fillna(-1).astype(int)

# Print the updated DataFrame
print(X)
print(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Select the top 10 features based on ANOVA F-statistic
k_best = SelectKBest(score_func=f_classif, k=20)
X_train_selected = k_best.fit_transform(X_train, y_train)
X_test_selected = k_best.transform(X_test)  # Use the same k_best for transforming the test data


print(y_test['Grade'].value_counts())
# Assuming X_train has column names
selected_feature_indices = k_best.get_support(indices=True)
selected_feature_names = X_train.columns[selected_feature_indices]

#Getting only wanted feature columns
X_train_selected = X_train[selected_feature_names]
X_test_selected = X_test[selected_feature_names]

print("Selected Feature Names:")
print(selected_feature_names)

print("Selected Feature Indices:")
print(selected_feature_indices)

# Number of selected features
num_selected_features = len(selected_feature_indices)

# Create scaler
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))

# Scale feature
X_train_selected_scaled = minmax_scale.fit_transform(X_train_selected) 
X_test_selected_scaled = minmax_scale.fit_transform(X_test_selected) 

# Build neural network using a sequential model
model = tf.keras.models.Sequential([
    # Add the input and first hidden layer
    tf.keras.layers.Dense(30, input_shape=(num_selected_features,), activation="sigmoid"),
    # Add the second hidden layer
    tf.keras.layers.Dense(15, activation="sigmoid"),
    # Add the output layer with the correct number of nodes
    tf.keras.layers.Dense(2, activation="softmax")
])

# Compile the model with the correct output layer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print the model summary
model.summary()

X_train_selected_scaled_df = pd.DataFrame(X_train_selected_scaled, columns=selected_feature_names)
X_test_selected_scaled_df = pd.DataFrame(X_test_selected_scaled, columns=selected_feature_names)

early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
model.fit(X_train_selected, y_train, epochs=5000, validation_data=(X_test_selected, y_test), callbacks=[early_stopping])
#model.fit(X_train_selected, y_train, epochs=5000) #5000



# Evaluate the model on the test data using evaluate
print("Evaluate on test data")
results = model.evaluate(X_test_selected, y_test)
print("test loss, test acc:", results)

#Saving the model
import os
path = os.path.dirname(os.path.realpath(__file__))
print("current directory:", path)
import joblib 
joblib.dump(model, path+'\\tumors_model.pkl')
print("Model dumped!")
# Save the k_best object

joblib.dump(k_best, path+'\\''k_best.pkl')
print("k best dumped dumped!")

model_columns = list(X_train_selected_scaled_df.columns)
print(model_columns)
joblib.dump(model_columns, path+'\\tumors_model_columns.pkl')
print("Models columns dumped!")

# Make predictions on the test data
predictions = model.predict(X_test_selected)

# Convert the predictions to class labels
predicted_labels = predictions.argmax(axis=1)

# Print the predicted labels
print("Predicted Labels:")
print(predicted_labels)

user_input = {
    'Age_at_diagnosis': 53.65, 
     'Race': 0, 
     'IDH1': 0, 
     'TP53': 1, 
     'ATRX': 1, 
     'PTEN': 1, 
     'EGFR': 1, 
     'CIC': 1, 
     'MUC16': 1, 
     'PIK3CA': 0, 
     'NF1': 1, 
     'PIK3R1': 1, 
     'FUBP1': 0, 
     'RB1': 1, 
     'NOTCH1': 0, 
     'CSMD3': 1, 
     'SMARCA4': 0, 
     'GRIN2A': 1, 
     'IDH2': 0, 
     'PDGFRA': 1
}

# Create a DataFrame from the user input
user_input_df = pd.DataFrame([user_input])
print(X_test_selected)

# Select only the relevant features
#user_input_selected = user_input_df[selected_feature_names]
# Create a DataFrame from the user input with the correct column names
user_input_df = pd.DataFrame([user_input], columns=selected_feature_names)

# Scale the input features using the same scaler used during training
user_input_scaled = minmax_scale.transform(user_input_df)

# Scale the input features using the same scaler used during training
# = minmax_scale.transform(user_input_selected)
# Make predictions
user_predictions = model.predict(user_input_scaled)

# Convert the predictions to class labels
user_predicted_label = np.argmax(user_predictions)



print("User Predictions label:", user_predicted_label)
print("User Predictions:", user_predictions)

for data in X_test_selected.values:
    print(data)
    # Select only the relevant features
    #user_input_selected = user_input_df[selected_feature_names]
    # Create a DataFrame from the user input with the correct column names
    user_input_df = pd.DataFrame([data], columns=selected_feature_names)
    
    # Scale the input features using the same scaler used during training
    user_input_scaled = minmax_scale.transform(user_input_df)
    
    # Scale the input features using the same scaler used during training
    # = minmax_scale.transform(user_input_selected)
    # Make predictions
    user_predictions = model.predict(user_input_scaled)
    
    # Convert the predictions to class labels
    user_predicted_label = np.argmax(user_predictions)
    
    

    print("User Predictions label:", user_predicted_label)
    print("User Predictions:", user_predictions)
    # Check if user_predicted_label is 1 and break the loop
    if user_predicted_label == 1:
       print("Breaking the loop because user_predicted_label is 1")
       break
