import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/cleaned/master.csv', low_memory=False)

# Print exact columns so you know what exists
print("All columns:", df.columns.tolist())

# Safe fill — only fill columns that actually exist
fill_zero = ['criminalcases', 'assets', 'liabilities']
fill_median = ['age', 'total_electors', 'turnout_percent', 'total_votes']
fill_unknown = ['education', 'category', 'symbol']

for col in fill_zero:
    if col in df.columns:
        df[col] = df[col].fillna(0)

for col in fill_median:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

for col in fill_unknown:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

# Drop useless columns
drop_cols = [
    'candidate_name', "father_husband's_name", 'photo_link',
    'address', 'application_date', 'application_status',
    'symbol', 'name', 'phase_y', 'sl_no',
    'total_electors_2019', 'age_2019', 'constituency_no'
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in ['state', 'constituency', 'constituency_matched', 'party',
            'gender', 'education', 'category', 'party_2019',
            'gender_2019', 'pc_name']:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

print("Shape after preprocessing:", df.shape)
print("Nulls remaining:", df.isnull().sum().sum())
print("Columns:", df.columns.tolist())

df.to_csv('data/cleaned/master.csv', index=False)
print("Saved.")