# Housing_Dataset.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('./dataset/Housing.csv')

# Basic Exploration
print(data.info())
print(data.describe())
print("Shape:", data.shape)

# Check for duplicates
duplicates = data.duplicated().sum()
print("Duplicates:", duplicates)
data.drop_duplicates(inplace=True)

# Categorical Unique Values
categorical_cols = data.select_dtypes(include='object').columns
for col in categorical_cols:
    print(f"{col}: {data[col].unique()}")

# Correlation matrix
corr_matrix = data.select_dtypes(include='number').corr()
print("\nCorrelation Matrix:\n", corr_matrix)

# Mean price by categorical feature
for col in categorical_cols:
    print(f"\nAverage price by {col}:")
    print(data.groupby(col)['price'].mean())

# Visualizations
sns.set(style='whitegrid')

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('images/correlation_heatmap.png')
plt.show()

# Average Price by Furnishing Status
plt.figure(figsize=(8,5))
sns.barplot(x='furnishingstatus', y='price', data=data)
plt.title('Average Price by Furnishing Status')
plt.savefig('images/price_furnishing.png')
plt.show()

# Distribution of House Prices
plt.figure(figsize=(8,5))
sns.histplot(data['price'], bins=30, kde=True)
plt.title('Distribution of House Prices')
plt.savefig('images/price_distribution.png')
plt.show()

# Price vs Main Road Access
plt.figure(figsize=(8,5))
sns.boxplot(x='mainroad', y='price', data=data)
plt.title('Price Distribution Based on Main Road Access')
plt.savefig('images/price_mainroad.png')
plt.show()
