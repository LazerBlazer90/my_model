# ----------------------------------------------------------------------
# Hugging Face Spaces App (app.py)
# This file serves both the HTML interface and the Prediction API.
# ----------------------------------------------------------------------

from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import os
import time

# --- Configuration ---
MODEL_FILE = 'model.joblib'
CLASS_LABELS = ['Setosa', 'Versicolor', 'Virginica']  # For the Iris model

app = Flask(__name__)

# --- Model Loading (Happens ONCE when the app starts) ---
try:
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model file '{MODEL_FILE}' not found. Please run 'create_model.py' first.")

    # Use the dummy model file we created
    ai_model = joblib.load(MODEL_FILE)
    print(f"Successfully loaded model from {MODEL_FILE}")
except Exception as e:
    ai_model = None
    print(f"Error loading model: {e}")
    print("The API will not function until the model is correctly saved.")


# --- BACKEND API ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    """Receives JSON data, makes a prediction, and returns the result."""
    if not ai_model:
        return jsonify({'error': 'Model not loaded on server.'}), 503

    try:
        # 1. Get JSON data from the request
        data = request.get_json()

        # Expecting a list of 4 features (sepal length, sepal width, etc.)
        features = data.get('features')

        if not features or len(features) != 4:
            return jsonify({'error': 'Invalid input: features must be a list of 4 numerical values.'}), 400

        # 2. Convert to NumPy array for the model (ensure it's a 2D array)
        input_array = np.array([features], dtype=np.float32)

        # 3. Make Prediction
        prediction_index = ai_model.predict(input_array)[0]

        # 4. Map index to a human-readable label
        predicted_label = CLASS_LABELS[prediction_index]

        # 5. Return the result as JSON
        return jsonify({
            'success': True,
            'prediction_index': int(prediction_index),
            'predicted_label': predicted_label,
            'timestamp': int(time.time())
        })

    except Exception as e:
        # Generic error handling
        return jsonify({
            'success': False,
            'error': f'Prediction failed due to server error: {str(e)}'
        }), 500


# --- FRONTEND (HTML, CSS, JS) ---

# Flask route to serve the main page
@app.route('/')
def index():
    """Serves the main HTML page."""

    # We use render_template_string to keep everything in one file for this example
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Deployment Demo (Iris)</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #f7f7f7; }
        .card { box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 0px 10px -5px rgba(0, 0, 0, 0.04); }
        input[type="number"]:focus { border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.5); }
    </style>
</head>
<body class="min-h-screen flex items-center justify-center p-4">

    <div class="card w-full max-w-lg bg-white p-8 rounded-xl">
        <h1 class="text-3xl font-extrabold text-gray-900 mb-6 text-center">Iris Flower Classifier</h1>
        <p class="text-gray-600 mb-8 text-center">
            Enter the four measurements below to predict the species (Setosa, Versicolor, or Virginica) using the backend model API.
        </p>

        <div id="loading-spinner" class="hidden text-center text-blue-500 mb-4">
            <svg class="animate-spin h-6 w-6 mx-auto" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <p class="mt-2 text-sm">Predicting...</p>
        </div>

        <form id="predictionForm" class="space-y-4">
            <!-- Input Fields -->
            <div class="grid grid-cols-2 gap-4">
                <div>
                    <label for="sepal_length" class="block text-sm font-medium text-gray-700">Sepal Length (cm)</label>
                    <input type="number" step="0.1" id="sepal_length" value="5.1" required class="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-3 border">
                </div>
                <div>
                    <label for="sepal_width" class="block text-sm font-medium text-gray-700">Sepal Width (cm)</label>
                    <input type="number" step="0.1" id="sepal_width" value="3.5" required class="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-3 border">
                </div>
                <div>
                    <label for="petal_length" class="block text-sm font-medium text-gray-700">Petal Length (cm)</label>
                    <input type="number" step="0.1" id="petal_length" value="1.4" required class="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-3 border">
                </div>
                <div>
                    <label for="petal_width" class="block text-sm font-medium text-gray-700">Petal Width (cm)</label>
                    <input type="number" step="0.1" id="petal_width" value="0.2" required class="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-3 border">
                </div>
            </div>

            <button type="submit" id="submitButton" class="w-full py-3 px-4 border border-transparent rounded-md shadow-lg text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition duration-150 ease-in-out">
                Get Prediction
            </button>
        </form>

        <!-- Output Display -->
        <div id="resultContainer" class="mt-8 pt-6 border-t border-gray-200">
            <h2 class="text-xl font-semibold text-gray-800 mb-3">Model Result:</h2>
            <div id="predictionOutput" class="text-lg font-bold text-gray-900 bg-gray-50 p-4 rounded-lg border border-gray-100">
                Awaiting input...
            </div>
            <div id="errorOutput" class="text-sm text-red-600 mt-2 hidden"></div>
        </div>
    </div>

    <!-- The critical JavaScript section that handles API communication -->
    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // Get elements
            const form = e.target;
            const submitButton = document.getElementById('submitButton');
            const loadingSpinner = document.getElementById('loading-spinner');
            const predictionOutput = document.getElementById('predictionOutput');
            const errorOutput = document.getElementById('errorOutput');

            // 1. Gather input data
            const features = [
                parseFloat(form.sepal_length.value),
                parseFloat(form.sepal_width.value),
                parseFloat(form.petal_length.value),
                parseFloat(form.petal_width.value)
            ];

            // 2. Prepare UI for request
            submitButton.disabled = true;
            submitButton.textContent = "Processing...";
            loadingSpinner.classList.remove('hidden');
            predictionOutput.textContent = "Sending data to model...";
            errorOutput.classList.add('hidden');

            try {
                // 3. Send data to the Flask API endpoint using fetch
                // IMPORTANT: Since the Flask app serves the page, the API path is relative.
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ features: features })
                });

                // 4. Handle HTTP errors
                if (!response.ok) {
                    const errorJson = await response.json();
                    throw new Error(errorJson.error || `HTTP error! Status: ${response.status}`);
                }

                // 5. Process successful response
                const result = await response.json();
                
                if (result.success) {
                    // Update the prediction display
                    predictionOutput.classList.remove('text-red-600');
                    predictionOutput.classList.add('text-green-600');
                    predictionOutput.innerHTML = `
                        <span class="text-2xl">${result.predicted_label}</span> 
                        <span class="text-gray-500 text-sm">(Class Index: ${result.prediction_index})</span>
                    `;
                } else {
                    // Handle API internal error structure
                    throw new Error(result.error || 'An unknown error occurred in the API.');
                }

            } catch (error) {
                // 6. Handle network or parsing errors
                console.error("Fetch error:", error);
                predictionOutput.classList.remove('text-green-600');
                predictionOutput.classList.add('text-red-600');
                predictionOutput.textContent = "Prediction Failed";
                errorOutput.textContent = `Error: ${error.message}`;
                errorOutput.classList.remove('hidden');
            } finally {
                // 7. Reset UI state
                submitButton.disabled = false;
                submitButton.textContent = "Get Prediction";
                loadingSpinner.classList.add('hidden');
            }
        });
    </script>

</body>
</html>
    """
    return render_template_string(html_content)


if __name__ == '__main__':
    # Flask default port, often handled by Hugging Face Spaces environment
    app.run(host='0.0.0.0', port=5000)
