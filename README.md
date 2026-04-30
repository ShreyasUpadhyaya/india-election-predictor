# India Election Predictor

Predict Lok Sabha election outcomes by constituency using machine learning.
Built on 2019 and 2024 election data — candidate profile, voter turnout, party, criminal record, and more.

---

## Live Demo

Coming soon — deploying on Render

---

## What It Does

User selects a state and constituency, fills in candidate details — age, party, education, criminal cases, expected turnout — and the model predicts whether that candidate wins or loses with a confidence score.

---

## Project Phases

- [x] Phase 1 — Web Portal UI
- [x] Phase 2 — Data Pipeline
- [x] Phase 3 — ML Model
- [x] Phase 4 — Flask API connected to model

---

## Tech Stack

| Layer | Tool |
| --- | --- |
| Frontend | HTML, CSS, JavaScript |
| Backend | Flask (Python) |
| ML  | Scikit-learn, Random Forest |
| Explainability | SHAP |
| Data Processing | Pandas, NumPy, thefuzz |
| Deployment | Render.com |

---

## Data Sources

| File | Source | Rows | Key Columns |
| --- | --- | --- | --- |
| candidates_2024.csv | Kaggle — jainaru/lok-sabha-2024-results-and-candidates | ~8000 | State, Constituency, Party, Age, Gender |
| turnout_2024.csv | data.opencity.in/dataset/parliament-elections-2024-voter-turnout | 543 | PC_Name, Turnout%, Total Electors |
| results_2019.csv | Kaggle — prakrutchauhan/indian-candidates-for-general-election-2019 | ~8000 | Winner, Party, Criminal Cases, Education |

---

## How to Run Locally

```bash
# Clone repo
git clone https://github.com/ShreyasUpadhyaya/india-election-predictor.git
cd india-election-predictor

# Install dependencies
pip install -r requirements.txt

# Run pipeline in order
python src/merge.py
python src/preprocessing.py
python src/model.py

# Start web app
python app/app.py
```

Open http://localhost:5000

---

## Project Structure

```markdown
india-election-predictor/
│
├── data/
│   ├── raw/
│   │   ├── candidates_2024.csv
│   │   ├── turnout_2024.csv
│   │   └── results_2019.csv
│   └── cleaned/
│       └── master.csv
│
├── src/
│   ├── merge.py            ← merges 3 files with fuzzy matching
│   ├── eda.py              ← exploratory analysis and charts
│   ├── preprocessing.py    ← cleans, encodes, engineers features
│   ├── model.py            ← trains Random Forest, saves model
│   └── predict.py          ← loads model, runs single prediction
│
├── app/
│   ├── app.py              ← Flask API with /predict endpoint
│   ├── templates/
│   │   └── index.html      ← web portal UI
│   └── static/
│       └── style.css
│
├── models/
│   ├── election_model.pkl  ← trained Random Forest
│   └── scaler.pkl          ← fitted StandardScaler
│
├── docs/
│   ├── notes.md
│   ├── feature_importance.png
│   ├── shap_summary.png
│   ├── criminal_vs_win.png
│   └── turnout_vs_margin.png
│
├── Procfile                ← for Render deployment
├── requirements.txt
└── README.md
```

``## Data Pipeline  **Step 1 — merge.py** Loads all 3 CSV files and standardizes column names. Fuzzy string matching via `thefuzz` resolves constituency name mismatches across datasets. 497 out of 540 constituencies matched directly. 43 resolved via fuzzy matching at 85% threshold.  **Step 2 — preprocessing.py** Drops useless columns: name, photo, address, application details. Fills nulls: criminal cases → 0, age → median, education → Unknown. Encodes categoricals: state, party, education, category via LabelEncoder. Drops leaky columns: vote counts that reveal the answer directly.  **Step 3 — model.py** Trains Random Forest on 8 clean features. 5-fold cross validation for reliable accuracy estimate. SHAP values for explainability. Saves model and scaler as pickle files.``

## Features Used

| Feature | Type | Why |
| --- | --- | --- |
| age | Numeric | Candidate age affects voter perception |
| constituency_matched | Encoded | Constituency identity |
| total_electors | Numeric | Constituency size |
| turnout_percent | Numeric | Voter engagement signal |
| total_votes | Numeric | Absolute vote volume |
| state | Encoded | Regional patterns differ significantly |
| category | Encoded | SC/ST/General reservation affects candidate field |
| education | Encoded | Candidate education level |

---

## Results

| Model | CV Accuracy |
| --- | --- |
| Random Forest (200 trees) | 68.3% ± 10.1% |

---

## Key Findings

- Turnout % and total electors are the strongest predictors in current dataset
- Criminal cases and education data was sparse — 4000+ null rows after merge — would improve model with cleaner data
- Fuzzy string matching was necessary — constituency names differ between ECI, Kaggle, and state sources
- Model deliberately excludes raw vote count columns to prevent data leakage

---

## Known Limitations

- 68% accuracy reflects data sparsity not model weakness
- Criminal cases, assets, and education missing for 4042 rows after merge
- Constituency and state are encoded integers — model cannot generalise to unseen names
- Adding caste demographics and alliance strength expected to push accuracy to 78–82%

---

## What Would Improve This

- Census 2011 caste demographic data per constituency
- Alliance strength feature: NDA vs INDIA seat share in state
- Incumbent flag: whether same candidate contested last election
- Margin from previous election as numeric feature
- XGBoost instead of Random Forest — handles missing values natively
- Larger dataset: merge 2009, 2014, 2019, 2024 for 2000+ constituency-level rows

---

## Author

Shreyas Upadhyaya
GitHub: https://github.com/ShreyasUpadhyaya
