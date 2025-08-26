import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load clean dataset
df = pd.read_csv('draft-data-20-years-college-stats-clean.csv')

df['College_3P%'] = df['College_3P%'].fillna(0)
df = df.dropna()

# Set features and target
X = df[['College_PPG', 'College_RPG', 'College_APG', 'College_FG%', 'College_3P%', 'College_FT%', 'College_MPG', 'College_STL', 'College_BLK', 'College_TOV']]
y = df['Success']

player_names = df['Player']

# Split
X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(X, y, player_names, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

intercept = model.intercept_[0]
coefficients = model.coef_[0]

# Predict and Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Logistic Regression Test Accuracy: {accuracy:.2%}")

# After model.fit(...)
feature_names = ['College_PPG', 'College_RPG', 'College_APG', 'College_FG%', 'College_3P%', 'College_FT%', 'College_MPG', 'College_STL', 'College_BLK', 'College_TOV']
coefficients = model.coef_[0]

# Create DataFrame
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

# Separate positive and negative coefficients
coef_pos = coef_df[coef_df['Coefficient'] > 0].sort_values(by='Coefficient', ascending=False)
coef_neg = coef_df[coef_df['Coefficient'] < 0].sort_values(by='Coefficient', ascending=False)

# Combine: positives first, negatives second
coef_final = pd.concat([coef_pos, coef_neg]).reset_index(drop=True)

# Show it
print(coef_final[['Feature', 'Coefficient']])

import matplotlib.pyplot as plt

# Plot
plt.figure(figsize=(10,6))
plt.barh(coef_final['Feature'], coef_final['Coefficient'], color=(coef_final['Coefficient'] > 0).map({True: 'green', False: 'red'}))
plt.xlabel('Coefficient')
plt.title('College Stats Importance for NBA Rookie Success')
plt.gca().invert_yaxis()  # largest coefficient on top
plt.show()

# After model.fit() and model.predict()

# Predict probabilities
y_prob = model.predict_proba(X_test_scaled)
success_prob = y_prob[:, 1]

# Attach to test set
results = pd.DataFrame({
    'Predicted_Success_Prob': success_prob,
    'Actual_Success': y_test.reset_index(drop=True)
})

# Sort by predicted probability
results = pd.DataFrame({
    'Player': names_test.reset_index(drop=True),
    'Predicted_Success_Prob': success_prob,
    'Actual_Success': y_test.reset_index(drop=True)
})

results = results.sort_values(by='Predicted_Success_Prob', ascending=True)

print(results.head(20))

import joblib
from google.colab import files

# --- Save model and scaler to disk ---
joblib.dump(model, 'nba_logistic_model.pkl')
joblib.dump(scaler, 'nba_scaler.pkl')

print("✅ Saved model and scaler as 'nba_logistic_model.pkl' and 'nba_scaler.pkl'")
