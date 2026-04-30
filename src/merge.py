import pandas as pd
from thefuzz import process  

# Install if needed: pip install thefuzz python-Levenshtein

file1 = pd.read_csv('data/raw/candidates_2024.csv')
file2 = pd.read_csv('data/raw/turnout_2024.csv')
file3 = pd.read_csv('data/raw/results_2019.csv')

# Standardize columns
for df in [file1, file2, file3]:
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Standardize constituency names
file1['constituency'] = file1['constituency'].str.lower().str.strip()
file2['pc_name'] = file2['pc_name'].str.lower().str.strip()
file3['constituency'] = file3['constituency'].str.lower().str.strip()

# Fuzzy match file1 constituencies to file3 constituencies
f3_names = file3['constituency'].unique().tolist()

def fuzzy_match(name, choices, threshold=85):
    match, score = process.extractOne(name, choices)
    return match if score >= threshold else name

file1['constituency_matched'] = file1['constituency'].apply(
    lambda x: fuzzy_match(x, f3_names)
)

file2['constituency_matched'] = file2['pc_name'].apply(
    lambda x: fuzzy_match(x, f3_names)
)

# Merge file1 + file2 on matched constituency
merged = pd.merge(
    file1, file2,
    left_on='constituency_matched',
    right_on='constituency_matched',
    how='left'
)

# Merge with file3
merged = pd.merge(
    merged, file3,
    left_on='constituency_matched',
    right_on='constituency',
    how='left'
)

print("Shape after merge:", merged.shape)
print("Nulls:\n", merged.isnull().sum())

merged.to_csv('data/cleaned/master.csv', index=False)
print("Saved to data/cleaned/master.csv")