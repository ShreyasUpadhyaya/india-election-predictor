import pandas as pd

# Load files
file1 = pd.read_csv('data/raw/candidates_2024.csv')
file2 = pd.read_csv('data/raw/turnout_2024.csv')
file3 = pd.read_csv('data/raw/results_2019.csv')

# Clean column names
def clean_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace('\n', '', regex=True)
        .str.replace('%', 'percent')
        .str.replace('/', '_')
        .str.replace('(', '')
        .str.replace(')', '')
    )
    return df

file1 = clean_columns(file1)
file2 = clean_columns(file2)
file3 = clean_columns(file3)

# Rename columns
file2 = file2.rename(columns={
    'pc_name': 'constituency',
    'polled_percent': 'turnout_percent'
})

# Clean text for merging
def clean_text(df):
    df['state'] = df['state'].str.strip().str.lower()
    df['constituency'] = df['constituency'].str.strip().str.lower()
    return df

file1 = clean_text(file1)
file2 = clean_text(file2)
file3 = clean_text(file3)

# Merge
merged = pd.merge(file1, file2, on=['state', 'constituency'], how='left')
merged = pd.merge(merged, file3, on=['state', 'constituency'], how='left', suffixes=('', '_2019'))

# Save
merged.to_csv('data/cleaned/master.csv', index=False)