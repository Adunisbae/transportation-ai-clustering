import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Sets up the visualisation comparison process
print("Part 4: Visualization Comparison")

# Defines file paths
data_dir = 'C:/Users/csmaogun/OneDrive - Liverpool John Moores University/6219COMP_Technology/AI Technique/Transportation_Dataset'
processed_dir = os.path.join(data_dir, 'processed')
visual_dir = 'visualizations'

# Create output directory
visual_dir = 'visualizations'
os.makedirs(visual_dir, exist_ok=True)

# Load both raw and preprocessed datasets for comparison
print("Loading raw and preprocessed datasets...")
try:
    raw_df = pd.read_csv(os.path.join(data_dir, 'combined_raw_uber_data.csv'))
    preprocessed_df = pd.read_csv(os.path.join(processed_dir, 'preprocessed_uber_data.csv'))
    print("Datasets loaded successfully!")
except FileNotFoundError:
    # Fall back step
    print("One or more dataset files not found. Generating synthetic data...")
    np.random.seed(42)
    n_samples = 2000
    latitudes = np.random.uniform(40.6, 40.9, n_samples)
    longitudes = np.random.uniform(-74.1, -73.7, n_samples)
    latitudes[:40] = np.random.uniform(39.0, 40.0, 40)
    longitudes[:40] = np.random.uniform(-75.0, -74.5, 40)
    months = np.array(['April'] * (n_samples // 2) + ['May'] * (n_samples // 2))
    np.random.shuffle(months)
    dates = [f"4/{np.random.randint(1, 31)}/2014 {np.random.randint(0,24)}:{np.random.randint(0,60)}:{np.random.randint(0,60)}" if m == 'April' 
             else f"5/{np.random.randint(1, 32)}/2014 {np.random.randint(0,24)}:{np.random.randint(0,60)}:{np.random.randint(0,60)}" for m in months]
    raw_df = pd.DataFrame({
        'Date/Time': dates,
        'Lat': latitudes,
        'Lon': longitudes,
        'Base': np.random.choice(['B02512', 'B02598', 'B02617', 'B02682', 'B02764'], n_samples),
        'month': months
    })
    preprocessed_df = raw_df[(raw_df['Lat'] >= 40.5) & (raw_df['Lat'] <= 41.0) & 
                              (raw_df['Lon'] >= -74.3) & (raw_df['Lon'] <= -73.7)].copy()
    preprocessed_df['Date/Time'] = pd.to_datetime(preprocessed_df['Date/Time'])
    preprocessed_df['hour'] = preprocessed_df['Date/Time'].dt.hour
    preprocessed_df['day_of_week'] = preprocessed_df['Date/Time'].dt.dayofweek
    preprocessed_df['day_of_month'] = preprocessed_df['Date/Time'].dt.day
    preprocessed_df['month_num'] = preprocessed_df['Date/Time'].dt.month

# Ensures datetime and derived columns are available in both datasets
raw_df['Date/Time'] = pd.to_datetime(raw_df['Date/Time'])
if 'hour' not in raw_df.columns:
    raw_df['hour'] = raw_df['Date/Time'].dt.hour
    raw_df['day_of_week'] = raw_df['Date/Time'].dt.dayofweek

# Creates visual comparisons between raw and preprocessed data
print("Creating visual comparisons...")

# Compares spatial distribution by month
print("\nCreating spatial distribution comparison by month...")
plt.figure(figsize=(15, 7))
for i, df in enumerate([raw_df, preprocessed_df]):
    plt.subplot(1, 2, i+1)
    plt.scatter(df[df['month'] == 'April']['Lon'], df[df['month'] == 'April']['Lat'], s=5, alpha=0.5, label='April', c='blue')
    plt.scatter(df[df['month'] == 'May']['Lon'], df[df['month'] == 'May']['Lat'], s=5, alpha=0.5, label='May', c='green')
    plt.title(f"{'Raw' if i == 0 else 'Cleaned'} Data - Spatial Distribution")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.savefig(f"{visual_dir}/spatial_distribution_comparison_by_month.png")
plt.show()
plt.close()

# Compares hourly distribution by month
print("Creating hourly distribution comparison by month...")
plt.figure(figsize=(15, 7))
for i, df in enumerate([raw_df, preprocessed_df]):
    plt.subplot(1, 2, i+1)
    for month, color in zip(['April', 'May'], ['blue', 'green']):
        sns.histplot(df[df['month'] == month]['hour'], bins=24, kde=True, color=color, alpha=0.5, label=month)
    plt.title(f"{'Raw' if i == 0 else 'Cleaned'} Data - Hourly Distribution")
    plt.xlabel('Hour of Day')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.savefig(f"{visual_dir}/hourly_distribution_comparison_by_month.png")
plt.show()
plt.close()

# Compares day of week distribution by month
print("Creating day of week distribution comparison by month...")
plt.figure(figsize=(15, 7))
for i, df in enumerate([raw_df, preprocessed_df]):
    plt.subplot(1, 2, i+1)
    sns.countplot(x='day_of_week', hue='month', data=df)
    plt.title(f"{'Raw' if i == 0 else 'Cleaned'} Data - Day of Week Distribution")
    plt.xlabel('Day of Week')
    plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.ylabel('Count')
    plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig(f"{visual_dir}/day_of_week_comparison_by_month.png")
plt.show()
plt.close()

# Compare data volume between raw and cleaned datasets by month
print("Creating data volume comparison bar chart...")
plt.figure(figsize=(10, 6))
volume = pd.DataFrame({
    'Raw': raw_df['month'].value_counts(),
    'Cleaned': preprocessed_df['month'].value_counts()
})
volume.plot(kind='bar', ax=plt.gca())
plt.title('Monthly Data Volume: Raw vs Cleaned')
plt.xlabel('Month')
plt.ylabel('Number of Records')
plt.grid(True, axis='y')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{visual_dir}/data_volume_comparison.png")
plt.show()
plt.close()

# Compares heatmaps  of hour against day of week by month
print("Creating heatmap comparison by month...")
plt.figure(figsize=(15, 12))
for i, (df, label, cmap) in enumerate([(raw_df, 'Raw', 'Blues'), (raw_df, 'Raw', 'Greens'),
                                      (preprocessed_df, 'Cleaned', 'Blues'), (preprocessed_df, 'Cleaned', 'Greens')]):
    plt.subplot(2, 2, i+1)
    month = 'April' if i % 2 == 0 else 'May'
    pivot = pd.crosstab(df[df['month'] == month]['day_of_week'], df[df['month'] == month]['hour'])
    sns.heatmap(pivot, cmap=cmap, annot=False, fmt='d')
    plt.title(f'{label} Data - {month} Pickup Heatmap')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.yticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.tight_layout()
plt.savefig(f"{visual_dir}/heatmap_comparison_by_month.png")
plt.show()
plt.close()

# Creates and saves summary statistics for both datasets
print("Creating summary statistics...")

def get_summary(df, label):
    # Groups by month and calculates statistics for key columns
    return df.groupby('month')[['Lat', 'Lon', 'hour', 'day_of_week']].describe().round(2)

# Get summary statistics
raw_summary = get_summary(raw_df, 'Raw')
cleaned_summary = get_summary(preprocessed_df, 'Cleaned')

#Save to CSV
raw_summary.to_csv(f"{visual_dir}/raw_summary_statistics_by_month.csv")
cleaned_summary.to_csv(f"{visual_dir}/preprocessed_summary_statistics_by_month.csv")

# Display raw data samples
print("\n===== RAW DATA =====")
for month in sorted(raw_df['month'].unique()):
    print(f"\n--- {month} Month - First 5 Rows ---")
    # Get the first 5 rows for this month and reset index to show row numbers
    month_data = raw_df[raw_df['month'] == month].head()
    # Format to match your example output
    print(month_data.to_string(index=True))

# Display cleaned data samples
print("\n\n===== CLEANED DATA =====")
for month in sorted(preprocessed_df['month'].unique()):
    print(f"\n--- {month} Month - First 5 Rows ---")
    # Get the first 5 rows for this month and reset index to show row numbers
    month_data = preprocessed_df[preprocessed_df['month'] == month].head()
    # Format to match your example output
    print(month_data.to_string(index=True))

# Also display summary statistics in a more compact format
print("\n\n===== SUMMARY STATISTICS =====")
print("\nRaw Data Statistics:")
for month in sorted(raw_df['month'].unique()):
    print(f"\n{month} Month - Key Statistics:")
    for col in ['Lat', 'Lon', 'hour', 'day_of_week']:
        stats = raw_summary.loc[month, col][['count', 'mean', 'std', 'min', 'max']]
        print(f"{col}: count={stats['count']:.0f}, mean={stats['mean']:.4f}, std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")

print("\nCleaned Data Statistics:")
for month in sorted(preprocessed_df['month'].unique()):
    print(f"\n{month} Month - Key Statistics:")
    for col in ['Lat', 'Lon', 'hour', 'day_of_week']:
        stats = cleaned_summary.loc[month, col][['count', 'mean', 'std', 'min', 'max']]
        print(f"{col}: count={stats['count']:.0f}, mean={stats['mean']:.4f}, std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")
        
print("Summary statistics saved and displayed.")

# Visualizes the effect of removing the outlier
print("Creating outlier comparison maps...")
plt.figure(figsize=(15, 10))

nyc_bounds = {'lat_min': 40.5, 'lat_max': 41.0, 'lon_min': -74.3, 'lon_max': -73.7}
raw_outliers = raw_df[(raw_df['Lat'] < nyc_bounds['lat_min']) | (raw_df['Lat'] > nyc_bounds['lat_max']) |
                      (raw_df['Lon'] < nyc_bounds['lon_min']) | (raw_df['Lon'] > nyc_bounds['lon_max'])]
raw_inliers = raw_df[(raw_df['Lat'] >= nyc_bounds['lat_min']) & (raw_df['Lat'] <= nyc_bounds['lat_max']) &
                     (raw_df['Lon'] >= nyc_bounds['lon_min']) & (raw_df['Lon'] <= nyc_bounds['lon_max'])]

# Before cleaning
plt.subplot(2, 1, 1)
plt.scatter(raw_inliers['Lon'], raw_inliers['Lat'], s=5, alpha=0.5, c='gray', label='Inliers')
plt.scatter(raw_outliers['Lon'], raw_outliers['Lat'], s=20, alpha=0.8, c='red', label='Outliers')
plt.title('Raw Data - Outliers Highlighted')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)

# After cleaning
plt.subplot(2, 1, 2)
plt.scatter(preprocessed_df['Lon'], preprocessed_df['Lat'], s=5, alpha=0.5, c='blue', label='Cleaned')
plt.title('Cleaned Data - Outliers Removed')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{visual_dir}/outliers_removal_comparison_by_month.png")
plt.show()
plt.close()

print("\nVisualization comparison complete! All charts saved to the 'visualizations' directory.")
