# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 19:42:00 2025

@author: aditya
Description: 
    EDA, Classification, and Regression on Indian Movie Dataset
"""

# 📚 Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dtale as dt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score

# 📥 Load Dataset
df = pd.read_csv(r"C:\Users\adity\Dropbox\PC\Documents\MTECH\SEM2\Python For Data Science\Project CA2\new_project\movies_updated.csv")

# 🔍 Interactive Data Viewer (D-Tale)
data = dt.show(df)
data.open_browser()

# 📊 Basic Info
print("Dataset Info:")
print(df.info())
print(df.describe())
print(df.head())

# 🔧 Missing Value Check
print("\nMissing values per column:")
print(df.isnull().sum())

# 🧹 Data Cleaning
df.columns = df.columns.str.strip().str.replace(',', '')  # Clean column names
df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')  # Convert to numeric if needed
df['rating'] = df['rating'].fillna('Not Rated')
df['writer'] = df['writer'].fillna('Unknown')
df['star'] = df['star'].fillna('Unknown')
df['company'] = df['company'].fillna('Unknown')
df['gross'] = df['gross'].fillna(df['gross'].median())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# 📌 Set Plot Style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 📈 EDA Visualizations

# Histogram of movie scores
sns.histplot(df['score'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Movie Scores')
plt.xlabel('Score')
plt.savefig('score_distribution.png')
plt.show()

# Votes vs Gross Earnings
sns.scatterplot(data=df, x='votes', y='gross', hue='rating')
plt.title('Votes vs Gross Earnings')
plt.xlabel('Votes')
plt.ylabel('Gross')
plt.savefig('votes_vs_gross.png')
plt.show()

# Budget vs Gross Earnings
sns.scatterplot(data=df, x='budget', y='gross', hue='rating')
plt.title('Budget vs Gross Earnings')
plt.xlabel('Budget')
plt.ylabel('Gross')
plt.savefig('budget_vs_gross.png')
plt.show()

# Correlation Heatmap
corr = df[['budget', 'gross', 'votes', 'score']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()

# Average Score by Rating
sns.barplot(data=df, x='rating', y='score', estimator='mean', errorbar=None, color='lightcoral')
plt.title('Average Movie Score by Rating')
plt.xlabel('Rating')
plt.ylabel('Average Score')
plt.xticks(rotation=45)
plt.savefig('avg_score_by_rating.png')
plt.show()

# Movie Count by Rating
sns.countplot(data=df, x='rating', color='#ff7f0e', edgecolor='black')
plt.title('Number of Movies by Rating', fontsize=14, pad=10)
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.savefig('count_by_rating.png', dpi=300, bbox_inches='tight')
plt.show()

# 🎯 Classification: Predicting Movie Rating
X = df[['budget', 'gross', 'votes', 'score']]  # Features
y = df['rating']  # Target

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_c, y_train_c)
y_pred_c = clf.predict(X_test_c)

# Classification Metrics
acc_class = accuracy_score(y_test_c, y_pred_c)
report_class = classification_report(y_test_c, y_pred_c, target_names=le.classes_)

print(f"\n🎯 Classification Accuracy: {acc_class:.4f}")
print("Classification Report:\n", report_class)

# 📉 Regression: Predicting Gross Earnings
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, df['gross'], test_size=0.2, random_state=42)

# Linear Regression Model
reg = LinearRegression()
reg.fit(X_train_r, y_train_r)
y_pred_r = reg.predict(X_test_r)

# Regression Metrics
mse = mean_squared_error(y_test_r, y_pred_r)
r2 = r2_score(y_test_r, y_pred_r)

print(f"\n📉 Regression MSE: {mse:.2f}")
print(f"📈 Regression R² Score: {r2:.4f}")

# Actual vs Predicted Gross Plot
sns.scatterplot(x=y_test_r, y=y_pred_r, alpha=0.5)
plt.xlabel('Actual Gross')
plt.ylabel('Predicted Gross')
plt.title('Actual vs Predicted Gross (Regression)')
plt.savefig('actual_vs_predicted_gross.png')
plt.show()
