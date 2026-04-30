from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

with open('models/election_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

FEATURES = ['age', 'constituency_matched', 'total_electors',
            'turnout_percent', 'total_votes', 'state', 'category', 'education']

# Encoding maps — match exactly what LabelEncoder used during preprocessing
# These are placeholder integer values — update with real encoded values after
# running: df[col].value_counts() in preprocessing.py
STATE_MAP = {
    'Uttar Pradesh': 28, 'Maharashtra': 15, 'West Bengal': 29,
    'Bihar': 4, 'Tamil Nadu': 25, 'Rajasthan': 22,
    'Madhya Pradesh': 14, 'Karnataka': 11, 'Gujarat': 8, 'Delhi': 6
}

CATEGORY_MAP = {'General': 0, 'SC': 1, 'ST': 2}

EDUCATION_MAP = {
    'Post Graduate': 5, 'Graduate': 3, 'Graduate Professional': 4,
    '12th Pass': 1, '10th Pass': 0, 'Literate': 2, 'Unknown': 6
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        input_dict = {
            'age': int(data['age']),
            'constituency_matched': int(data['constituency_matched']),
            'total_electors': int(data['total_electors']),
            'turnout_percent': float(data['turnout_percent']),
            'total_votes': int(data['total_votes']),
            'state': STATE_MAP.get(data['state'], 0),
            'category': CATEGORY_MAP.get(data['category'], 0),
            'education': EDUCATION_MAP.get(data['education'], 6)
        }

        df = pd.DataFrame([input_dict])[FEATURES]
        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0]

        return jsonify({
            'prediction': int(prediction),
            'win_probability': round(float(probability[1]) * 100, 1),
            'lose_probability': round(float(probability[0]) * 100, 1),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

# With this
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)