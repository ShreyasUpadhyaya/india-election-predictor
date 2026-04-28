import pandas as pd

df = pd.read_csv('data/cleaned/master.csv', low_memory=False)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
)

df = df.rename(columns={
    'criminalcases': 'criminal_cases'
})

df['criminal_cases'] = df['criminal_cases'].fillna(0)
df['age'] = df['age'].fillna(df['age'].median())
df['age_2019'] = df['age_2019'].fillna(df['age_2019'].median())
df['turnout_percent'] = df['turnout_percent'].fillna(df['turnout_percent'].median())

df['gender'] = df['gender'].fillna('unknown')
df['party'] = df['party'].fillna('unknown')
df['education'] = df['education'].fillna('unknown')

df['criminal_cases'] = pd.to_numeric(df['criminal_cases'], errors='coerce').fillna(0)
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['age_2019'] = pd.to_numeric(df['age_2019'], errors='coerce')

df.to_csv('data/cleaned/final_dataset.csv', index=False)