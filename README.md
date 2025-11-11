# Uber Pickup Demand Clustering Analysis

## Overview
Analyzed 1.2M+ Uber pickup records from April-May 2014 using unsupervised machine learning to identify geographic demand hotspots in New York City. This project demonstrates spatial clustering, temporal pattern analysis, and data-driven insights for transportation optimization.

**Key Result:** Identified 5 distinct geographic clusters with silhouette score of **0.463**, each representing different urban functional zones (Midtown, Financial District, Residential, Outer Boroughs, Transportation Hubs).

## Problem Statement
Urban transportation systems generate massive datasets that can reveal patterns for:
- **Infrastructure Planning**: Identifying high-demand zones for resource allocation
- **Surge Pricing Optimization**: Understanding temporal and spatial demand patterns
- **Driver Allocation**: Strategic positioning based on predictable hotspots

This project uncovers these patterns using clustering algorithms on real-world ride-sharing data.

## Dataset
- **Source**: Kaggle (FiveThirtyEight) - Uber Pickups in NYC
- **Time Period**: April-May 2014
- **Records**: 1,216,951 pickup records
- **Features**: Date/Time, Latitude, Longitude, Uber Base Code
- **Geographic Coverage**: All 5 boroughs of New York City

## Methodology

### Data Preprocessing
- **Temporal Features**: Extracted hour, day of week, day of month from timestamps
- **Outlier Removal**: Removed geographic outliers outside NYC bounds (40.5-41.0°N, -74.3--73.7°W)
  - Removed: ~5,000 records (~0.4%)
  - Preserved: 1,212,397 clean records
- **Feature Scaling**: StandardScaler normalization for clustering algorithms
- **Feature Sets**: Created 3 variants:
  - Spatial only (Lat/Lon)
  - Spatial + Temporal (Lat/Lon/Hour/DayOfWeek)
  - Spatial + Temporal + Month (for seasonal comparison)

### Clustering Algorithms

#### K-means Clustering
- **Optimal k**: 5 (determined via Elbow Method + Silhouette Score analysis)
- **Parameters**: `n_clusters=5, random_state=42, n_init=10`
- **Performance**: Silhouette Score = **0.463** ✓
- **Advantage**: Well-defined clusters with interpretable centroids

#### DBSCAN Clustering
- **Parameters**: `eps=0.1, min_samples=15`
- **Performance**: Silhouette Score = 0.212
- **Advantage**: Captures irregular cluster shapes, identifies noise points (18% outliers)
- **Use Case**: Validates K-means results, reveals sparse pickup zones

## Key Findings

### Geographic Clusters (K-means)

| Cluster | Location | Characteristics | Peak Hours | Pattern |
|---------|----------|-----------------|------------|---------|
| 0 | Midtown Manhattan | Business district, shopping, tourism | 8-9 AM, 5-7 PM | Weekday peaks |
| 1 | Lower Manhattan/Financial District | Government, business hubs | 9 AM-6 PM | Business hours |
| 2 | Upper Manhattan/Upper East Side | Residential, museums, Central Park | Evening & weekends | Evening peaks |
| 3 | Outer Boroughs (Brooklyn/Queens) | Residential, nightlife | 6-7 AM, 3-7 PM | Early morning & late night |
| 4 | Transportation Hubs | JFK, LaGuardia airports | Flight arrival/departure times | Event-driven |

### Temporal Patterns
- **Morning Rush (7-9 AM)**: Dominates Clusters 0-1 (business districts)
- **Evening Rush (5-7 PM)**: Highest activity across all clusters
- **Weekday vs Weekend**: Business clusters (0-1) peak weekdays; Residential clusters (2-3) peak weekends
- **Month Comparison**: May showed 8% higher volume than April, particularly on weekends

### Noise Analysis (DBSCAN)
- 18% of records classified as noise (isolated pickups)
- Concentrated in outer boroughs and edges of Manhattan
- Validates that core clusters capture 82% of mainstream demand

## Results

### Visualizations Generated
- Spatial distribution maps (before/after cleaning)
- Hourly & daily distribution analysis
- Temporal heatmaps by hour and day of week
- Cluster comparison (K-means vs DBSCAN)
- Silhouette score analysis

### Business Recommendations
1. **Driver Allocation**: Direct 40% of drivers to Clusters 0-1 during business hours
2. **Surge Pricing**: Implement 15-20% premium during 5-7 PM across all clusters
3. **Weekend Strategy**: Increase driver availability in Clusters 2-3 on Saturdays/Sundays
4. **Airport Optimization**: Cluster 4 requires specialized scheduling based on flight times

## Technical Stack
- **Language**: Python 3.x
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (K-means, DBSCAN)
- **Visualization**: Matplotlib, Seaborn
- **Metrics**: Silhouette Score, Davies-Bouldin Index

## Project Structure
\`\`\`
├── README.md
├── data_exploration.py          # Initial data loading & analysis
├── data_preprocessing.py        # Cleaning, feature extraction, scaling
├── clustering_process.py        # K-means & DBSCAN implementation
├── visualization_comparison.py  # Before/after preprocessing comparisons
└── visualizations/
    ├── spatial_distribution.png
    ├── hourly_distribution.png
    ├── daily_distribution.png
    ├── hour_day_heatmap.png
    ├── kmeans_elbow_silhouette.png
    ├── kmeans_clusters.png
    ├── dbscan_clusters.png
    ├── cluster_0_heatmap_by_month.png
    └── [5 additional cluster heatmaps...]
\`\`\`

## How to Run

### 1. Install Dependencies
\`\`\`bash
pip install pandas numpy scikit-learn matplotlib seaborn
\`\`\`

### 2. Download Dataset
Download from Kaggle: [Uber Pickups in NYC](https://www.kaggle.com/datasets/fivethirtyeight/uber-pickups-in-new-york-city)

### 3. Run Analysis Pipeline
\`\`\`bash
# Step 1: Explore raw data
python data_exploration.py

# Step 2: Preprocess & clean
python data_preprocessing.py

# Step 3: Apply clustering algorithms
python clustering_process.py

# Step 4: Compare results
python visualization_comparison.py
\`\`\`

### 4. View Results
All visualizations saved in `/visualizations` folder

## Key Learnings

### What Worked Well
- K-means silhouette score of 0.463 indicates reasonable cluster separation
- Temporal patterns clearly align with real-world NYC transportation behavior
- Combining K-means + DBSCAN provides complementary insights

### Challenges & Solutions
- **High dimensionality**: Solved by standardizing features to unit variance
- **Outlier sensitivity**: K-means vulnerable to outliers; mitigated with preprocessing
- **Parameter selection**: Tested eps values 0.05-0.2; eps=0.1 provided best balance

### If I Repeated This Project
- Include weather data (temperature, rain) as additional features
- Implement Hierarchical Clustering for dendrogram visualization
- Deploy interactive Plotly dashboard for stakeholder exploration
- Add real-time prediction capability using Prophet for forecasting

## Relevant Coursework & Skills Demonstrated
- Applied Data Science (ML algorithms, unsupervised learning)
- Advanced AI (clustering techniques, optimization)
- Data Visualization (interpreting complex patterns)
- Statistical Analysis (evaluation metrics, hypothesis validation)

---

**Grade**: First Class (University of Liverpool, 2025)  
**Dataset**: 1.2M+ real-world transportation records  
**Duration**: 4-week project  
**Contact**: [Your Email] | [Your LinkedIn]
