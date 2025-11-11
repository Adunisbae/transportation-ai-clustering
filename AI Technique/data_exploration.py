import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Loads and explores the April and May Uber datasets 
print("Part 1: Data Exploration")

# Create a directory for visualizations if it doesn't exist
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Load April and May Uber datasets
print("Loading April and May datasets...")
try:
    df_april = pd.read_csv('C:/Users/csmaogun/OneDrive - Liverpool John Moores University/6219COMP_Technology/AI Technique/Transportation_Dataset/uber-raw-data-apr14.csv')
    df_may = pd.read_csv('C:/Users/csmaogun/OneDrive - Liverpool John Moores University/6219COMP_Technology/AI Technique/Transportation_Dataset/uber-raw-data-may14.csv')
    
    df_april['month'] = 'April'
    df_may['month'] = 'May'
    
    # Combine datasets
    df = pd.concat([df_april, df_may], ignore_index=True)
    print(f"Datasets loaded and combined successfully! Total records: {len(df)}")
    
except FileNotFoundError:
    print("One or both dataset files not found. Please check file paths.")
    raise

# Overview of the datasets
print("\nDataset Overview:")
print(df.info())

# First 5 rows of each month's data
print("\nFirst 5 rows of April dataset:")
print(df[df['month'] == 'April'].head())

print("\nFirst 5 rows of May dataset:")
print(df[df['month'] == 'May'].head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Summary statistics per month
print("\nApril Dataset Statistics:")
print(df[df['month'] == 'April'][['Lat', 'Lon']].describe())

print("\nMay Dataset Statistics:")
print(df[df['month'] == 'May'][['Lat', 'Lon']].describe())

# Creates temporal features
print("\nCreating temporal features...")

df['Date/Time'] = pd.to_datetime(df['Date/Time'])
df['hour'] = df['Date/Time'].dt.hour
df['day_of_week'] = df['Date/Time'].dt.dayofweek

# Creates different visualization to represent the raw data
print("Generating visualizations...")

# 1. Spatial distribution - shows where pickups occur geographically
plt.figure(figsize=(12, 10))
plt.scatter(df[df['month'] == 'April']['Lon'], df[df['month'] == 'April']['Lat'], s=5, c='blue', alpha=0.5, label='April')
plt.scatter(df[df['month'] == 'May']['Lon'], df[df['month'] == 'May']['Lat'], s=5, c='green', alpha=0.5, label='May')
plt.title('Spatial Distribution of Uber Pickups (April & May)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.savefig('visualizations/spatial_distribution.png')
plt.show()
plt.close()

# 2. Hourly distribution - shows when pickups occur throughpout the day per month 
plt.figure(figsize=(14, 6))
for i, month in enumerate(['April', 'May']):
    plt.subplot(1, 2, i+1)
    hourly = df[df['month'] == month]['hour'].value_counts().sort_index()
    sns.barplot(x=hourly.index, y=hourly.values, color='skyblue')
    plt.title(f'Pickups by Hour - {month}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Pickups')
    plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('visualizations/hourly_distribution.png')
plt.show()
plt.close()

# 3. Daily distribution - shows patterns across days of the week for each month
plt.figure(figsize=(14, 6))
for i, month in enumerate(['April', 'May']):
    plt.subplot(1, 2, i+1)
    daily = df[df['month'] == month]['day_of_week'].value_counts().sort_index()
    sns.barplot(x=daily.index, y=daily.values, color='salmon')
    plt.title(f'Pickups by Day of Week - {month}')
    plt.xlabel('Day of Week (0=Mon, 6=Sun)')
    plt.ylabel('Number of Pickups')
    plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('visualizations/daily_distribution.png')
plt.show()
plt.close()

# 4. Heatmaps: Hour x Day of Week - shows temporal patterns
plt.figure(figsize=(16, 10))
for i, month in enumerate(['April', 'May']):
    plt.subplot(2, 1, i+1)
    pivot = pd.crosstab(df[df['month'] == month]['day_of_week'], df[df['month'] == month]['hour'])
    sns.heatmap(pivot, cmap='YlGnBu')
    plt.title(f'Heatmap of Pickups by Hour and Day - {month}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.yticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
plt.tight_layout()
plt.savefig('visualizations/hour_day_heatmap.png')
plt.show()
plt.close()

# 5. Base distribution - shows pickup patterns by Uber base
plt.figure(figsize=(14, 6))
for i, month in enumerate(['April', 'May']):
    plt.subplot(1, 2, i+1)
    base_counts = df[df['month'] == month]['Base'].value_counts()
    sns.barplot(x=base_counts.index, y=base_counts.values, palette='viridis')
    plt.title(f'Pickups by Base - {month}')
    plt.xlabel('Base')
    plt.ylabel('Number of Pickups')
    plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('visualizations/base_distribution.png')
plt.show()
plt.close()

# Saves tbe final comnbined dataset
output_path = 'C:/Users/csmaogun/OneDrive - Liverpool John Moores University/6219COMP_Technology/AI Technique/Transportation_Dataset/combined_raw_uber_data.csv'
df.to_csv(output_path, index=False)
print(f"\nCombined data saved to: {output_path}")
print("Data exploration complete. All visualizations saved in the 'visualizations' folder.")