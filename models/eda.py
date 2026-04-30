import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/cleaned/master.csv')

print(df.shape)
print(df.describe())

# 1. Win rate by party
if 'party' in df.columns and 'winner' in df.columns:
    win_rate = df.groupby('party')['winner'].mean().sort_values(ascending=False)
    print("\nTop 10 win rates by party:")
    print(win_rate.head(10))

# 2. Does criminal record affect winning?
if 'criminal_cases' in df.columns:
    criminal_win = df.groupby('winner')['criminal_cases'].mean()
    print("\nAvg criminal cases — winners vs losers:")
    print(criminal_win)
    
    sns.boxplot(x='winner', y='criminal_cases', data=df[df['criminal_cases'] < 20])
    plt.title('Criminal Cases: Winners vs Losers')
    plt.xticks([0, 1], ['Lost', 'Won'])
    plt.savefig('docs/criminal_vs_win.png')
    plt.show()

# 3. Turnout vs margin
if 'polled_(%)' in df.columns and 'margin_pct' in df.columns:
    sns.scatterplot(x='polled_(%)', y='margin_pct', data=df, alpha=0.4)
    plt.title('Turnout vs Win Margin')
    plt.savefig('docs/turnout_vs_margin.png')
    plt.show()

# 4. Gender win rate
if 'gender' in df.columns:
    gender_win = df.groupby('gender')['winner'].mean()
    print("\nWin rate by gender:")
    print(gender_win)

# 5. Education vs winning
if 'education' in df.columns:
    edu_win = df.groupby('education')['winner'].mean().sort_values(ascending=False)
    print("\nWin rate by education:")
    print(edu_win)