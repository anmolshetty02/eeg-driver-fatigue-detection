import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

print("Loading EEG dataset...")
df = pd.read_csv('data/eeg_combined.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nClass distribution:")
print(df['label'].value_counts())

X = df.drop('label', axis=1).values
y = df['label'].values

print(f"\nFeatures: {X.shape}")
print(f"Labels: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

print("\n1. Random Forest Classifier...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_train)

rf_pred = rf.predict(X_test_scaled)
rf_proba = rf.predict_proba(X_test_scaled)[:, 1]

rf_acc = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_proba)

print(f"   Accuracy: {rf_acc:.3f} ({rf_acc*100:.1f}%)")
print(f"   AUC-ROC: {rf_auc:.3f}")

print("\n2. Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]

lr_acc = accuracy_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_proba)

print(f"   Accuracy: {lr_acc:.3f} ({lr_acc*100:.1f}%)")
print(f"   AUC-ROC: {lr_auc:.3f}")

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"Random Forest  - Accuracy: {rf_acc*100:.1f}%, AUC: {rf_auc:.3f}")
print(f"Logistic Reg   - Accuracy: {lr_acc*100:.1f}%, AUC: {lr_auc:.3f}")

print("\n" + "="*60)
print("RANDOM FOREST - DETAILED METRICS")
print("="*60)
print(classification_report(y_test, rf_pred, target_names=['Alert', 'Drowsy']))

cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Alert', 'Drowsy'],
            yticklabels=['Alert', 'Drowsy'])
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=300)
print("\nConfusion matrix saved to results/confusion_matrix.png")

feature_names = df.drop('label', axis=1).columns.tolist()
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.title('Feature Importance for Drowsiness Detection')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=300)
print("Feature importance plot saved to results/feature_importance.png")

print("\nTop 5 Features:")
for i in range(5):
    idx = indices[i]
    print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.3f}")

with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nModel and scaler saved to models/")
print("\n✅ Training complete!")
