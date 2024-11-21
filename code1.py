#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

# Check the data structure
print(df.head())
print(df.info())

# Rename columns for easier access
df.columns = [
    'Country', 'WPSI_2023', 'WDI_Total_2019', 'WDI_Street_Safety_2019', 
    'WDI_Intentional_Homicide_2019', 'WDI_NonPartner_Violence_2019',
    'WDI_IntimatePartner_Violence_2019', 'WDI_Legal_Discrimination_2019', 
    'WDI_Global_Gender_Gap_2019', 'WDI_Gender_Inequality_2019', 
    'WDI_AttitudesTowardViolence_2019'
]
# Check for missing values
print(df.isnull().sum())

# Drop rows with any missing values if needed
df_cleaned = df.dropna()

# Option 2: Impute missing values with the mean for numeric columns only
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Basic descriptive statistics to get an overview of each numeric column
print(df[numeric_cols].describe())


# Correlation matrix
corr_matrix = df[numeric_cols].corr()

# Visualize the correlation matrix with a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Women Danger Indicators')
plt.show()

# Top 10 most dangerous countries based on the WDI Total Score
top_dangerous = df.sort_values(by='WDI_Total_2019', ascending=False).head(10)
print("Top 10 most dangerous countries for women based on WDI Total Score:\n", top_dangerous[['Country', 'WDI_Total_2019']])

# Top 10 safest countries based on the WDI Total Score
bottom_dangerous = df.sort_values(by='WDI_Total_2019', ascending=True).head(10)
print("\nTop 10 safest countries for women based on WDI Total Score:\n", bottom_dangerous[['Country', 'WDI_Total_2019']])

# Display the column names to check for any discrepancies
print("Column names in DataFrame:", df.columns)

# Select only numeric columns
numeric_df = df.select_dtypes(include=[float, int])

# Calculate correlation matrix on numeric data only
correlation_matrix = numeric_df.corr()

# Display correlation matrix
print("Correlation matrix:\n", correlation_matrix)


# Histogram of WDI_Total_2019 scores
plt.figure(figsize=(10, 6))
sns.histplot(df['WDI_Total_2019'], bins=20, kde=True, color='coral')
plt.title("Distribution of WDI Total Score (2019)")
plt.xlabel("WDI Total Score")
plt.ylabel("Frequency")
plt.show()

# Bar plot for Top 10 most dangerous countries
plt.figure(figsize=(12, 6))
sns.barplot(x='WDI_Total_2019', y='Country', data=top_dangerous, palette='Reds_r')
plt.title("Top 10 Most Dangerous Countries for Women (WDI Total Score 2019)")
plt.xlabel("WDI Total Score")
plt.ylabel("Country")
plt.show()

# Bar plot for Top 10 safest countries
plt.figure(figsize=(12, 6))
sns.barplot(x='WDI_Total_2019', y='Country', data=bottom_dangerous, palette='Greens')
plt.title("Top 10 Safest Countries for Women (WDI Total Score 2019)")
plt.xlabel("WDI Total Score")
plt.ylabel("Country")
plt.show()


# Plotting the distribution of WDI Street Safety
plt.figure(figsize=(10, 6))
sns.histplot(df['WDI_Street_Safety_2019'], kde=True, color='green')
plt.title("Distribution of Street Safety Scores (WDI 2019)")
plt.xlabel("Street Safety Score (WDI 2019)")
plt.ylabel("Frequency")
plt.show()



# Create a new column for composite danger score
df['Composite_Danger_Score'] = df[[
    'WDI_Total_2019',
    'WDI_Street_Safety_2019',
    'WDI_Intentional_Homicide_2019',
    'WDI_NonPartner_Violence_2019',
    'WDI_IntimatePartner_Violence_2019',
    'WDI_Legal_Discrimination_2019',
    'WDI_Global_Gender_Gap_2019',
    'WDI_Gender_Inequality_2019',
    'WDI_AttitudesTowardViolence_2019'
]].mean(axis=1)

# Sort by the composite score
top_composite = df.sort_values(by='Composite_Danger_Score', ascending=False).head(10)
bottom_composite = df.sort_values(by='Composite_Danger_Score', ascending=True).head(10)

print("Top 10 countries by Composite Danger Score:\n", top_composite[['Country', 'Composite_Danger_Score']])
print("\nTop 10 safest countries by Composite Danger Score:\n", bottom_composite[['Country', 'Composite_Danger_Score']])



# Top 10 countries with the highest rates of intentional homicide
top_homicide = df.sort_values(by='WDI_Intentional_Homicide_2019', ascending=False).head(10)
print("Top 10 countries with the highest intentional homicide rate:\n", top_homicide[['Country', 'WDI_Intentional_Homicide_2019']])

# Top 10 countries with the lowest rates of intentional homicide
bottom_homicide = df.sort_values(by='WDI_Intentional_Homicide_2019', ascending=True).head(10)
print("\nTop 10 countries with the lowest intentional homicide rate:\n", bottom_homicide[['Country', 'WDI_Intentional_Homicide_2019']])

# Create a safety score by combining various columns (lower values are safer)
df['Safety_Score'] = (df['WDI_Street_Safety_2019'] + 
                       df['WDI_Intentional_Homicide_2019'] + 
                       df['WDI_NonPartner_Violence_2019'] + 
                       df['WDI_IntimatePartner_Violence_2019']) / 4
# Rank countries by the safety score (higher values are more dangerous)
df['Safety_Rank'] = df['Safety_Score'].rank(ascending=True)
df_sorted_by_safety = df[['Country', 'Safety_Score', 'Safety_Rank']].sort_values(by='Safety_Rank')

print("Countries ranked by safety:\n", df_sorted_by_safety[['Country', 'Safety_Score', 'Safety_Rank']])