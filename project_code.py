# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 19:42:00 2025

@author: adity
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dtale as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Data
df = pd.read_csv(r"C:\Users\adity\Dropbox\PC\Documents\MTECH\SEM2\Python For Data Science\Project CA2\new_project\movies_updated.csv")
data=dt.show(df)
data.open_browser()
print("Dataset Info:")
print(df.info())
print(df.describe())
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())
# Data Cleaning
df.columns = df.columns.str.strip().str.replace(',', '')
df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
df['rating'] = df['rating'].fillna('Not Rated')
df['writer'] = df['writer'].fillna('Unknown')
df['star'] = df['star'].fillna('Unknown')
df['company'] = df['company'].fillna('Unknown')
df['gross'] = df['gross'].fillna(df['gross'].median())
print(df.isnull().sum())
# Set styles
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# EDA Visualizations
sns.histplot(df['score'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Movie Scores')
plt.xlabel('Score')
plt.savefig('score_distribution.png')
plt.show()

sns.scatterplot(data=df, x='votes', y='gross', hue='rating')
plt.title('Votes vs Gross Earnings')
plt.xlabel('Votes')
plt.ylabel('Gross')
plt.savefig('votes_vs_gross.png')
plt.show()

sns.scatterplot(data=df, x='budget', y='gross', hue='rating')
plt.title('Budget vs Gross Earnings')
plt.xlabel('Budget')
plt.ylabel('Gross')
plt.savefig('budget_vs_gross.png')
plt.show()

corr = df[['budget', 'gross', 'votes', 'score']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()



sns.barplot(data=df, x='rating', y='gross', estimator='mean', errorbar=None, color='lightgreen')
plt.title('Average Gross Earnings by Rating')
plt.xlabel('Rating')
plt.ylabel('Average Gross')
plt.xticks(rotation=45)
plt.savefig('avg_gross_by_rating.png')
plt.show()


sns.countplot(data=df, x='rating', color='salmon')
plt.title('Number of Movies by Rating')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('count_by_rating.png')
plt.show()



sns.barplot(data=df, x='rating', y='score', estimator='mean', errorbar=None, color='lightcoral')
plt.title('Average Movie Score by Rating')
plt.xlabel('Rating')
plt.ylabel('Average Score')
plt.xticks(rotation=45)
plt.savefig('avg_score_by_rating.png')
plt.show()


sns.boxplot(data=df, x='rating', y='gross', color='lightblue')
plt.title('Gross Earnings Distribution by Rating')
plt.xlabel('Rating')
plt.ylabel('Gross')
plt.xticks(rotation=45)
plt.savefig('gross_box_by_rating.png')
plt.show()

sns.countplot(data=df, x='rating', color='#ff7f0e', edgecolor='black')
plt.title('Number of Movies by Rating', fontsize=14, pad=10)
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.savefig('count_by_rating.png', dpi=300, bbox_inches='tight')
plt.show()

#



# Classification Model
X = df[['budget', 'gross', 'votes', 'score']]
y = df['rating']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_c, y_train_c)
y_pred_c = clf.predict(X_test_c)

acc_class = accuracy_score(y_test_c, y_pred_c)
report_class = classification_report(y_test_c, y_pred_c, target_names=le.classes_)
print(f"Classification Accuracy: {acc_class:.4f}")
print("Classification Report:\n", report_class)

# Regression Model
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, df['gross'], test_size=0.2, random_state=42)
reg = LinearRegression()
reg.fit(X_train_r, y_train_r)
y_pred_r = reg.predict(X_test_r)

mse = mean_squared_error(y_test_r, y_pred_r)
r2 = r2_score(y_test_r, y_pred_r)
print(f"Regression MSE: {mse:.2f}")
print(f"Regression R² Score: {r2:.4f}")

sns.scatterplot(x=y_test_r, y=y_pred_r, alpha=0.5)
plt.xlabel('Actual Gross')
plt.ylabel('Predicted Gross')
plt.title('Actual vs Predicted Gross (Regression)')
plt.savefig('actual_vs_predicted_gross.png')
