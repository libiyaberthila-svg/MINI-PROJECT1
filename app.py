from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = [float(x) for x in request.form.values()]
        final = np.array(values).reshape(1, -1)
        prediction = model.predict(final)

        output = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        return render_template("index.html", prediction_text="Result: " + output)

    except:
        return render_template("index.html", prediction_text="Error in input")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
