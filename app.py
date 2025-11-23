from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# --------------------------
# Load Model
# --------------------------
model = pickle.load(open("LRModel.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # get input from HTML form
    feature1 = float(request.form['feature1'])
    feature2 = float(request.form['feature2'])
    
    # Convert to array
    input_data = np.array([[feature1, feature2]])

    # Predict
    result = model.predict(input_data)[0]

    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
