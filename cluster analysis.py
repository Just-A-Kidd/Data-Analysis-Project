#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
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

# Option 2: Impute missing values with the mean for numeric columns only
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Basic descriptive statistics to get an overview of each numeric column
print(df[numeric_cols].describe())

# Select the relevant columns for clustering (based on the variables available)
cluster_columns = ['WDI_Street_Safety_2019', 'WDI_Intentional_Homicide_2019', 
                   'WDI_NonPartner_Violence_2019', 'WDI_IntimatePartner_Violence_2019', 
                   'WDI_Legal_Discrimination_2019', 'WDI_Gender_Inequality_2019']

# Handle missing values (e.g., dropping rows with missing values for simplicity)
df_cleaned = df[cluster_columns].dropna()

# Normalize the data (important for K-means)
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df_cleaned)

# Create a new DataFrame with the normalized data (optional for better understanding)
df_normalized = pd.DataFrame(normalized_data, columns=cluster_columns)
# Elbow method to find the optimal number of clusters
inertia = []
k_range = range(1, 11)  # Try 1 to 10 clusters

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_normalized)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o', color='blue')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.show()
# Apply K-Means with the chosen number of clusters (e.g., 3 clusters)
kmeans = KMeans(n_clusters=6, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_normalized)

# Add the cluster labels back to the original DataFrame
df['Cluster'] = df['Cluster'].astype('category')

# Show the first few rows of the DataFrame with the cluster labels
print(df[['Country', 'Cluster']].head())

# Visualization of clusters using pairplot
sns.pairplot(df, hue='Cluster', vars=cluster_columns, palette="Set1")
plt.suptitle("Cluster Analysis of Countries Based on Safety Indicators", y=1.02)
plt.show()

# Alternatively, use a scatter plot (reduce to 2D using PCA or t-SNE for visualization)
from sklearn.decomposition import PCA

# Reduce the data to 2D for visualization
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_normalized)

# Create a DataFrame with the 2D components and cluster labels
df_pca = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
df_pca['Cluster'] = df['Cluster']

# Scatter plot of the first two principal components
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='Set1', s=100)
plt.title("PCA of Clustered Countries")
plt.show()
# Analyze the cluster centers to understand the characteristics of each cluster
cluster_centers = kmeans.cluster_centers_
cluster_centers_df = pd.DataFrame(cluster_centers, columns=cluster_columns)
print("Cluster Centers (means of each cluster):\n", cluster_centers_df)

# Check the distribution of countries across clusters
print(df['Cluster'].value_counts())