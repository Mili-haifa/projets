from flask import Flask, request, jsonify
import numpy as np
import pickle

# Load the saved model
with open('model/Iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a new Flask app
app = Flask(__name__)

# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json(force=True)

    # Extract the input data from the JSON object
    input_data = data['input']

    # Convert the input data to a numpy array
    input_array = np.array(input_data)

    # Use the loaded model to make predictions
    predictions = model.predict(input_array)

    # Convert the predictions to a list
    predictions_list = predictions.tolist()

    # Return the predictions as a JSON response
    return jsonify({'predictions': predictions_list})

   
# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)