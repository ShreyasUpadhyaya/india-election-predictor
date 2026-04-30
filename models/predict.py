import pandas as pd
import pickle

with open('models/election_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

FEATURES = ['age', 'constituency_matched', 'total_electors', 
            'turnout_percent', 'total_votes', 'state', 'category', 'education']

def predict_winner(input_dict):
    df = pd.DataFrame([input_dict])[FEATURES]
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0]
    return {
        'prediction': int(prediction),
        'win_probability': round(float(probability[1]) * 100, 1),
        'lose_probability': round(float(probability[0]) * 100, 1)
    }

# Test
sample = {
    'age': 52,
    'constituency_matched': 120,
    'total_electors': 1500000,
    'turnout_percent': 65.0,
    'total_votes': 975000,
    'state': 5,
    'category': 1,
    'education': 3
}

result = predict_winner(sample)
print("Prediction:", "WIN" if result['prediction'] == 1 else "LOSE")
print("Win probability:", result['win_probability'], "%")