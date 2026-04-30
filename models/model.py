import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/cleaned/master.csv', low_memory=False)

# Check nulls before doing anything
print("Nulls per column:\n", df.isnull().sum())

# Fill all nulls
df.fillna(df.median(numeric_only=True), inplace=True)
# Drop duplicate columns from merge
drop_cols = ['phase_y', 'total_electors_2019', 'age_2019', 'sl_no']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Verify nulls are gone
print("Nulls after fix:", df.isnull().sum().sum())

# Target = winner column (1 = won, 0 = lost)
df = df[df['winner'].notna()]
df['winner'] = df['winner'].astype(int)

# Drop leaky vote count columns
leaky = [
    'general\nvotes', 'postal\nvotes', 'total\nvotes',
    'over_total_electors_\nin_constituency',
    'over_total_votes_polled_\nin_constituency',
    'total_electors_y', 'age_y', 'pc_name'
]
df.drop(columns=[c for c in leaky if c in df.columns], inplace=True)

# Rename duplicates
df.rename(columns={
    'age_x': 'age',
    'total_electors_x': 'total_electors',
    'polled_(%)': 'turnout_percent'
}, inplace=True)

X = df.drop(columns=['winner'])
y = df['winner']

# Remove leaky columns — these reveal the answer
leaky_cols = [
    'generalvotes', 'postalvotes', 'totalvotes',
    'over_total_electors_in_constituency',
    'over_total_votes_polled_in_constituency',
    'constituency_no', 'phase_x', 'sl_no'
]
X = X.drop(columns=[c for c in leaky_cols if c in X.columns])
X = X.select_dtypes(include='number')
print("Clean features:", X.columns.tolist())

print("Features used:", X.columns.tolist())
print("Class distribution:\n", y.value_counts())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# Evaluate
print("\nRandom Forest:", rf.score(X_test_scaled, y_test))
y_pred = rf.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Cross validation
cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"\nCV Accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

# Feature importance
feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
feat_imp.sort_values().plot(kind='barh', figsize=(10, 6))
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('docs/feature_importance.png')
plt.show()

# SHAP
explainer = shap.TreeExplainer(rf)
shap_values = explainer(X_test_scaled)
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('docs/shap_summary.png')
plt.show()
print("FINAL FEATURES:", X.columns.tolist())
# Save model and scaler
with open('models/election_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nModel saved to models/election_model.pkl")
print("Scaler saved to models/scaler.pkl")

# AFTER selecting X
feature_names = X.columns.tolist()

with open('models/features.pkl', 'wb') as f:
    pickle.dump(feature_names, f)