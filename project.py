import joblib

# Replace "model.pkl" with the actual filename of your model
model = joblib.load("/Users/egesa/Downloads/FakeDetectionSystem/trained.pkl")

# Example data as a NumPy array
data = [[1, 2, 3], [4, 5, 6]]

# Example data as a pandas DataFrame
data = pd.DataFrame({
    "feature1": [1, 4],
    "feature2": [2, 5],
    "feature3": [3, 6]
})

# Select the features you want to use for prediction
features = data[["feature1", "feature2"]]

predictions = model.predict(features)

# Print the model to confirm it's loaded
print(predictions)