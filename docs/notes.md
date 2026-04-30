# India Election Predictor — Project Notes

## Problem Statement
Predict whether a candidate will win a Lok Sabha constituency
based on voter data, candidate profile, and historical results.

## Data Sources
| File | Source | Rows | Key Columns |
|---|---|---|---|
| candidates_2024.csv | Kaggle — jainaru/lok-sabha-2024-results-and-candidates | ~8000 | State, Constituency, Party, Age, Gender |
| turnout_2024.csv | data.opencity.in | 543 | PC_Name, Turnout%, Total Electors |
| results_2019.csv | Kaggle — prakrutchauhan/indian-candidates-for-general-election-2019 | ~8000 | Winner, Party, Criminal Cases, Education |

## Data Pipeline

### merge.py
- Loads all 3 CSV files
- Standardizes column names: lowercase, underscores
- Fuzzy matches constituency names using thefuzz at 85% threshold
- 497/540 constituencies matched directly
- 43 resolved via fuzzy matching
- Output: data/cleaned/master.csv

### preprocessing.py
- Drops useless columns: name, photo, address, application details
- Fills nulls: criminalcases → 0, age → median, education → Unknown
- Encodes categoricals via LabelEncoder: state, party, education, category
- Drops leaky columns: raw vote counts that reveal the answer
- Drops duplicate merge columns: phase_y, age_y, total_electors_2019
- Output: overwrites data/cleaned/master.csv

### model.py
- Loads cleaned master.csv
- Removes leaky features: generalvotes, postalvotes, totalvotes, vote share columns
- Trains Random Forest (200 trees) on 8 features
- 5-fold cross validation
- Saves SHAP summary plot and feature importance chart to docs/
- Saves model to models/election_model.pkl
- Saves scaler to models/scaler.pkl

## Features Used
| Feature | Type | Why |
|---|---|---|
| age | Numeric | Candidate age affects voter perception |
| constituency_matched | Encoded | Constituency identity |
| total_electors | Numeric | Constituency size |
| turnout_percent | Numeric | Voter engagement signal |
| total_votes | Numeric | Absolute vote volume |
| state | Encoded | Regional patterns differ significantly |
| category | Encoded | SC/ST/General reservation affects candidate field |
| education | Encoded | Candidate education level |

## Model Results
| Metric | Value |
|---|---|
| CV Accuracy | 68.3% |
| CV Std Dev | ± 10.1% |
| Algorithm | Random Forest |
| Trees | 200 |
| Train/Test Split | 80/20 |

## Web Application

### app/app.py
Flask API with two routes:
- `GET /` — serves index.html
- `POST /predict` — accepts JSON, returns prediction and probability

Input JSON format:
```json
{
  "age": 52,
  "constituency_matched": 120,
  "total_electors": 1500000,
  "turnout_percent": 65.0,
  "total_votes": 975000,
  "state": "Uttar Pradesh",
  "category": "General",
  "education": "Post Graduate"
}
```

Output JSON format:
```json
{
  "prediction": 1,
  "win_probability": 68.3,
  "lose_probability": 31.7,
  "status": "success"
}
```

### app/templates/index.html
Single page web portal with:
- State and constituency dropdowns
- Candidate profile inputs: party, gender, education, age, criminal cases, category
- Voter data inputs: turnout slider, incumbent toggle
- Predict button calling /predict endpoint via fetch POST
- Result card showing predicted outcome and probability bars

## Deployment
- Platform: Render.com
- Python version: 3.11.9
- Server: Gunicorn
- Start command: gunicorn app.app:app
- Environment variable: PORT=10000

## Known Limitations
- 68% accuracy reflects data sparsity not model weakness
- Criminal cases and education null for 4042 rows after merge
- constituency_matched and state are encoded integers — model cannot generalise to unseen names
- total_votes is borderline leaky — present election votes used as feature

## What Would Improve This
- Census 2011 caste demographic data per constituency
- Alliance strength: NDA vs INDIA seat share per state
- Incumbent flag: same candidate from last election
- Margin from previous election as numeric feature
- XGBoost — handles missing values natively, usually outperforms RF on tabular data
- Merge 2009 + 2014 data for larger training set

## Commit History
| Commit | Description |
|---|---|
| [INIT] | Folder structure, gitkeep placeholders, README |
| [UI] | Web portal with input form, dark theme, orange accent |
| [DATA] | merge.py — 3 files joined with fuzzy matching |
| [CLEAN] | preprocessing.py — nulls, encoding, feature engineering |
| [EDA] | eda.py — win rate, criminal cases, turnout analysis |
| [MODEL] | Random Forest, SHAP, cross validation, pickle save |
| [CONNECT] | Flask API connected to trained model |
| [DEPLOY] | Render deployment, Python 3.11, gunicorn config |