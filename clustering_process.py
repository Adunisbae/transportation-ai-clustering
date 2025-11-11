import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# Initializes the clustering process
print("Part 3: Clustering Process")

# Define file paths
data_dir = 'C:/Users/csmaogun/OneDrive - Liverpool John Moores University/6219COMP_Technology/AI Technique/Transportation_Dataset'
processed_dir = os.path.join(data_dir, 'processed')
visual_dir = 'visualizations'

if not os.path.exists(visual_dir):
    os.makedirs(visual_dir)
    
# Load the preprocessed data and scaled features
print("Loading preprocessed data and scaled features...")
try:
    df = pd.read_csv(os.path.join(processed_dir, 'preprocessed_uber_data.csv'))
    X_spatial_scaled = np.load(os.path.join(processed_dir, 'X_spatial_scaled.npy'))
    X_temporal_spatial_scaled = np.load(os.path.join(processed_dir, 'X_temporal_spatial_scaled.npy'))
    X_with_month_scaled = np.load(os.path.join(processed_dir, 'X_with_month_scaled.npy'))
    print("Preprocessed data loaded successfully!")
except FileNotFoundError:
    print("Preprocessed data not found. Please run the preprocessing step first.")
    raise

# Samples equally from each month to keep the comparison unbiased
df_apr = df[df['month'] == 'April'].sample(n=2500, random_state=42)
df_may = df[df['month'] == 'May'].sample(n=2500, random_state=42)
df = pd.concat([df_apr, df_may], ignore_index=True)

# Recalculate scaled features using the sampled dataset
from sklearn.preprocessing import StandardScaler
X_spatial_scaled = StandardScaler().fit_transform(df[['Lat', 'Lon']])
X_temporal_spatial_scaled = StandardScaler().fit_transform(df[['Lat', 'Lon', 'hour', 'day_of_week']])
X_with_month_scaled = StandardScaler().fit_transform(df[['Lat', 'Lon', 'hour', 'day_of_week', 'month_num']])

# Implements K-means clustering with optimal k selection
print("\n K-means Clustering")
k_range = range(2, 11)
inertia = []
silhouette_scores = []

# Uses both elbow method (inertia) and silhouette scores
for k in k_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_spatial_scaled)
    inertia.append(model.inertia_)
    if k > 1:
        silhouette_scores.append(silhouette_score(X_spatial_scaled, model.labels_))

# Creates visualization to help determine optimal k
# Elbow method - look for the 'bend' in the curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'bo-')
plt.title('Elbow Method')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.grid(True)

# Silhouette scores - higher is better
plt.subplot(1, 2, 2)
plt.plot(list(k_range), silhouette_scores, 'ro-')
plt.title('Silhouette Scores')
plt.xlabel('k')
plt.ylabel('Score')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{visual_dir}/kmeans_elbow_silhouette.png')
plt.show()
plt.close()

# Based on elbow method and silhouette score, k=5 seems optimal
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster_kmeans'] = kmeans.fit_predict(X_spatial_scaled)

# Visualize spatial clusters
plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, optimal_k))
for i in range(optimal_k):
    cluster = df[df['cluster_kmeans'] == i]
    plt.scatter(cluster['Lon'], cluster['Lat'], s=10, c=[colors[i]], label=f'Cluster {i}', alpha=0.5)
plt.title('K-means Spatial Clustering')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.legend()
plt.savefig(f'{visual_dir}/kmeans_clusters.png')
plt.show()
plt.close()

# Implements DBSCAN clustering
print("\n DBSCAN Clustering")

dbscan = DBSCAN(eps=0.1, min_samples=15)
df['cluster_dbscan'] = dbscan.fit_predict(X_spatial_scaled)

# Visualize DBSCAN custers
plt.figure(figsize=(12, 8))
labels = sorted(df['cluster_dbscan'].unique(), key=lambda x: (x == -1, x))
colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))

# Creates a custom order for the legend
legend_handles = []
legend_labels = []

# Plots all clusters to get the visualization
for i, label in enumerate(labels):
    cluster = df[df['cluster_dbscan'] == label]
    color = 'black' if label == -1 else colors[i]
    scatter = plt.scatter(cluster['Lon'], cluster['Lat'], s=10, c=[color], 
                         label=f'Cluster {label}', alpha=0.5)
    # Store handle and label for ordered legend
    legend_handles.append(scatter)
    legend_labels.append(f'Cluster {label}')

plt.title('DBSCAN Spatial Clustering')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)

# Create legend with custom order
plt.legend(handles=legend_handles, labels=legend_labels)
plt.savefig(f'{visual_dir}/dbscan_clusters.png')

# Evaluates clustering quality using silhouette scores
print("\n Evaluation")
scores = {
    'K-means': silhouette_score(X_spatial_scaled, df['cluster_kmeans']),
    'DBSCAN': silhouette_score(X_spatial_scaled[df['cluster_dbscan'] != -1], df['cluster_dbscan'][df['cluster_dbscan'] != -1])
    if len(set(df['cluster_dbscan'])) > 1 and -1 in df['cluster_dbscan'].values else 0
}

# Plot visualisation comparison
plt.figure(figsize=(8, 6))
plt.bar(scores.keys(), scores.values(), color=['blue', 'red'])
plt.title('Silhouette Score Comparison')
plt.ylabel('Score')
for i, score in enumerate(scores.values()):
    plt.text(i, score + 0.01, f'{score:.3f}', ha='center')
plt.ylim(0, max(scores.values()) * 1.2)
plt.grid(True, axis='y')
plt.savefig(f'{visual_dir}/silhouette_comparison.png')

# Creates temporal heatmaps for each cluster to analyze time patterns
print("\n Heatmaps by Cluster")
for cluster in range(optimal_k):
    plt.figure(figsize=(12, 10))
    for idx, month in enumerate(['April', 'May']):
        plt.subplot(2, 1, idx + 1)
        subset = df[(df['cluster_kmeans'] == cluster) & (df['month'] == month)]
        pivot = pd.crosstab(subset['day_of_week'], subset['hour'])
        sns.heatmap(pivot, cmap='coolwarm', cbar=True)
        plt.title(f'Cluster {cluster} - {month}')
        plt.xlabel('Hour')
        plt.ylabel('Day of Week')
        plt.yticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
    plt.tight_layout()
    plt.savefig(f'{visual_dir}/cluster_{cluster}_heatmap_by_month.png')
    plt.show()
    plt.close()

print("\nClustering analysis complete! Results saved in 'visualizations' folder.")
