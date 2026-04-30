# India Election Predictor — Project Notes

## Problem Statement
Predict whether a candidate will win a Lok Sabha constituency
based on voter data, candidate profile, and historical results.

## Data Sources
| File | Source | Rows | Key Columns |
|---|---|---|---|
| candidates_2024.csv | Kaggle | ~8000 | State, Constituency, Party, Age, Gender |
| turnout_2024.csv | data.opencity.in | 543 | PC_Name, Turnout%, Total Electors |
| results_2019.csv | Kaggle | ~8000 | Winner, Party, Criminal Cases, Education |

## Data Pipeline
1. merge.py — fuzzy matched constituency names across 3 files using thefuzz
   - 497/540 constituencies matched directly
   - 43 matched via fuzzy string matching at 85% threshold
2. preprocessing.py — cleaned nulls, encoded categoricals, dropped leaky columns
3. master.csv — final merged dataset: ~32000 rows, 8 features

## Features Used
| Feature | Type | Why |
|---|---|---|
| age | Numeric | Candidate age affects voter perception |
| constituency_matched | Encoded | Constituency identity |
| total_electors | Numeric | Constituency size |
| turnout_percent | Numeric | Voter engagement signal |
| total_votes | Numeric | Absolute vote volume |
| state | Encoded | Regional patterns differ |
| category | Encoded | SC/ST/General reservation affects field |
| education | Encoded | Candidate education level |

## Model
- Algorithm: Random Forest (200 trees)
- Train/Test Split: 80/20
- CV Accuracy: 68.3% ± 10.1%
- Evaluation: 5-fold cross validation

## Known Limitations
- 68% accuracy is ceiling with current features
- Criminal cases and assets data missing for 4000+ rows after merge
- constituency_matched and total_votes are proxy features not true predictors
- What would improve it: caste demographic data, alliance data, incumbent flag, margin from last election

## What I Would Do Next
- Add census demographic data per constituency
- Add alliance strength feature (NDA vs INDIA seat count in state)
- Add margin from previous election as feature
- Retrain with XGBoost and compare
- Add SHAP plots to web portal

## Tech Stack
| Layer | Tool |
|---|---|
| Data | Pandas, thefuzz |
| ML | Scikit-learn Random Forest |
| Explainability | SHAP |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Render.com |

## How to Run Locally
```bash
pip install -r requirements.txt
python src/merge.py
python src/preprocessing.py
python src/model.py
python app/app.py
```
Open http://localhost:5000

## Commit History
- [INIT] Folder structure and README
- [UI] Web portal with input form
- [DATA] Merge 3 source files with fuzzy matching
- [CLEAN] Preprocessing and feature engineering
- [MODEL] Random Forest training and SHAP analysis
- [CONNECT] Flask API connected to trained model