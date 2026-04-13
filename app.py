from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load ML model
model = pickle.load(open('PROJECTMODEL.pkl', 'rb'))

# Dummy user database (for demo)
users = {
    "admin@gmail.com": "1234",
    "user@gmail.com": "pass123"
}

# Home route
@app.route('/')
def home():
    return render_template('index.html')


# 🔴 LOGIN API
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')

        # Check user
        if email in users and users[email] == password:
            return jsonify({
                "status": "success",
                "message": "Login successful"
            })
        else:
            return jsonify({
                "status": "fail",
                "message": "Invalid email or password"
            })

    except Exception as e:
        return jsonify({"error": str(e)})


# 🔵 PREDICT API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        input_data = np.array([
            data['age'],
            data['sex'],
            data['cp'],
            data['trestbps'],
            data['chol'],
            data['fbs'],
            data['restecg'],
            data['thalach'],
            data['exang'],
            data['oldpeak'],
            data['slope'],
            data['ca'],
            data['thal']
        ]).reshape(1, -1)
        prediction = model.predict(input_data)[0]

        return jsonify({
            "result": int(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)