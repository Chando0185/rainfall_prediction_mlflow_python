from flask import Flask, request, render_template
import pandas as pd
import mlflow.sklearn

# Initialize Flask app
app = Flask(__name__)

# Load the production model
mlflow.set_experiment("Rainfall1")
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

model_version = 1
production_model_name = "rainfall-prediction-production"
prod_model_uri = f"models:/{production_model_name}@champion"
loaded_model = mlflow.sklearn.load_model(prod_model_uri)

# Feature names
feature_names = ['pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

@app.route('/')
def home():
    return render_template('index.html', prediction_result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        input_data = [
            float(request.form['pressure']),
            float(request.form['dewpoint']),
            float(request.form['humidity']),
            float(request.form['cloud']),
            float(request.form['sunshine']),
            float(request.form['winddirection']),
            float(request.form['windspeed']),
        ]

        # Create DataFrame for the input
        input_df = pd.DataFrame([input_data], columns=feature_names)

        # Predict using the loaded model
        prediction = loaded_model.predict(input_df)

        # Convert prediction to human-readable format
        result = "Rainfall" if prediction[0] == 1 else "No Rainfall"

        return render_template('index.html', prediction_result=result)
    except Exception as e:
        return render_template('index.html', prediction_result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=8080)
