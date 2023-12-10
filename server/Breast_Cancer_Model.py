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
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
Y = breast_cancer_wisconsin_diagnostic.data.targets 

print(X.head(10))

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Transform the target variable to 0 and 1
y=label_encoder.fit_transform(Y)
# Retrieve the original class labels
original_labels = label_encoder.classes_

# Print the mapping
print(f"Label 0 corresponds to: {original_labels[0]}")
print(f"Label 1 corresponds to: {original_labels[1]}")



print(y)

# print(y)
  
# # metadata 
# print(breast_cancer_wisconsin_diagnostic.metadata) 
  
# # variable information 
# print(breast_cancer_wisconsin_diagnostic.variables) 

# print('X')
# print(X.columns)
# print(X.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(y_test)

# Select the top 10 features based on ANOVA F-statistic
k_best = SelectKBest(score_func=f_classif, k=10)
X_train_selected = k_best.fit_transform(X_train, y_train)
X_test_selected = k_best.transform(X_test)



print(X_train_selected.shape)




# Assuming X_train has column names
selected_feature_indices = k_best.get_support(indices=True)
selected_feature_names = X_train.columns[selected_feature_indices]

#Getting only wanted feature columns
X_train_selected = X_train[selected_feature_names]
X_test_selected = X_test[selected_feature_names]

print(X_train_selected)

# print("Selected Feature Names:")
print(selected_feature_names)

print("Selected Feature Indices:")
print(selected_feature_indices)

print("X_train_selected:")
print(X_train_selected)


# Number of selected features
num_selected_features = len(selected_feature_indices)

# Create scaler
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
# Scale feature
X_train_selected_scaled = minmax_scale.fit_transform(X_train_selected) 
X_test_selected_scaled = minmax_scale.fit_transform(X_test_selected) 
print("X_train_selected:")
print(X_train_selected)

# # Create scaler
# standard_scaler = StandardScaler()

# # Scale features
# X_train_scaled = standard_scaler.fit_transform(X_train)
# X_test_scaled = standard_scaler.transform(X_test)



# Build neural network using a sequential model
model = tf.keras.models.Sequential([
    # Add the input and first hidden layer
    tf.keras.layers.Dense(30, input_shape=(num_selected_features,), activation="sigmoid"),
    # Add the second hidden layer
    tf.keras.layers.Dense(15, activation="sigmoid"),
    # Add the output layer
    tf.keras.layers.Dense(2, activation="sigmoid")
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print the model summary
model.summary()

print('X train selected')
print(X_train_selected)

X_train_selected_scaled_df = pd.DataFrame(X_train_selected_scaled, columns=selected_feature_names)
X_test_selected_scaled_df = pd.DataFrame(X_test_selected_scaled, columns=selected_feature_names)

print(X_train_selected_scaled_df)
print(X_test_selected_scaled_df)



early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
model.fit(X_train_selected_scaled_df, y_train, epochs=5000, validation_data=(X_test_selected_scaled_df, y_test), callbacks=[early_stopping])
#model.fit(X_train_selected, y_train, epochs=5000) #5000



# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(X_test_selected_scaled_df, y_test)
print("test loss, test acc:", results)

#Saving the model
import os
path = os.path.dirname(os.path.realpath(__file__))
print("current directory:", path)
import joblib 
joblib.dump(model, path+'\\model_breast_cancer.pkl')
print("Model dumped!")
# Save the k_best object

joblib.dump(k_best, path+'\\''k_best.pkl')
print("Models columns dumped!")

model_columns = list(X_train_selected_scaled_df.columns)
print(model_columns)
joblib.dump(model_columns, path+'\\model_breast_cancer_columns.pkl')
print("Models columns dumped!")


user_input={
  "radius1": 13.54,
  "perimeter1": 14.36,
  "area1": 87.46,
  "concavity1": 0.09779,
  "concave_points1": 0.08129,
  "radius3": 15.76,
  "perimeter3": 102.5,
  "area3": 764.0,
  "concavity3": 0.1234,
  "concave_points3": 0.0678

}

# Create a DataFrame from the user input
user_input_df = pd.DataFrame([user_input])

# Select only the relevant features
user_input_selected = user_input_df[selected_feature_names]

# Scale the input features using the same scaler used during training
user_input_scaled = minmax_scale.transform(user_input_selected)

# Make predictions
user_predictions = model.predict(user_input_scaled)
print(user_predictions)

# Convert the predictions to class labels
user_predicted_label = np.argmax(user_predictions)

print("User Predicted Label:", user_predicted_label)

print('DF COLUMN NAMES')
print(X_train_selected_scaled_df.columns)

print(model_columns)


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


