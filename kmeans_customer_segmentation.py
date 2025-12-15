# Component 4: K-Means Customer Segmentation
# Prepared as part of mentorship program
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



df = pd.DataFrame({
    'CustomerID': [101,102,103,104,105,106,107,108,109,110],
    'Recency': [10, 200, 30, 400, 15, 90, 300, 20, 60, 500],
    'Frequency': [20, 2, 15, 1, 25, 8, 3, 18, 10, 1],
    'Monetary': [5000, 300, 4000, 150, 6000, 1200, 500, 4500, 2000, 100]
})

X = df[['Recency', 'Frequency', 'Monetary']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

# choose optimal k after seeing elbow
optimal_k = 3

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)


plt.figure(figsize=(8,6))
plt.scatter(df['Frequency'], df['Monetary'], c=df['Cluster'])
plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.title('Customer Segments (Frequency vs Monetary)')
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(df['Recency'], df['Monetary'], c=df['Cluster'])
plt.xlabel('Recency')
plt.ylabel('Monetary')
plt.title('Customer Segments (Recency vs Monetary)')
plt.show()  

print("Mean values per cluster:") #mean calculate
print(df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean())

print("\nMedian values per cluster:")#median calculate
print(df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].median())

print("\nStandard deviation per cluster:")#mode calculate
print(df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].std())

df.groupby('Cluster')['Monetary'].mean().plot(kind='bar')
plt.xlabel('Cluster')
plt.ylabel('Average Monetary Value')
plt.title('Average Spending per Customer Segment')
plt.show()

df.boxplot(column='Frequency', by='Cluster')
plt.title('Purchase Frequency by Cluster')
plt.suptitle('')
plt.xlabel('Cluster')
plt.ylabel('Frequency')
plt.show()
