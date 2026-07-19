from flask import Flask, render_template, request
import pickle
import json

app = Flask(__name__)

# Load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load column order
with open("columns.json", "r") as f:
    columns = json.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    input_data = []

    for col in columns:
        value = request.form.get(col)
        input_data.append(float(value))

    final_input = scaler.transform([input_data])

    prob = model.predict_proba(final_input)[0][1]
    prediction = 1 if prob > 0.4 else 0

    if prediction == 1:
        result = f"⚠️ High Risk ({round(prob*100,2)}%)"
    else:
        result = f"✅ Safe ({round(prob*100,2)}%)"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
