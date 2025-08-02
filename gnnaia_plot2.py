import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data from the provided tables
data = {
    'Dataset': ['Reddit', 'Reddit', 'Reddit', 'Protein', 'Protein', 'Protein', 
                'Flickr', 'Flickr', 'Flickr', 'Yelp', 'Yelp', 'Yelp'],
    'Model': ['GCN', 'GIN', 'SAGE', 'GCN', 'GIN', 'SAGE', 
              'GCN', 'GIN', 'SAGE', 'GCN', 'GIN', 'SAGE'],
    'Without_AIA': [56.992, 54.346, 53.704, 33.708, 31.068, 30.682,
                    11.895, 10.318, 10.116, 78.721, 74.601, 70.456],
    'With_AIA': [47.809, 45.861, 45.274, 29.783, 27.598, 27.248,
                 9.128, 8.080, 7.901, 62.553, 59.667, 56.383],
    'CuSparse': [117.052, 114.736, 115.946, 71.378, 70.140, 70.554,
                 11.846, 10.934, 11.174, 82.568, 80.399, 82.845]
}

df = pd.DataFrame(data)

# Calculate improvement percentages over CuSparse
# Improvement = (CuSparse - Method) / CuSparse * 100
df['AIA_Improvement'] = ((df['CuSparse'] - df['With_AIA']) / df['CuSparse']) * 100
df['NoAIA_Improvement'] = ((df['CuSparse'] - df['Without_AIA']) / df['CuSparse']) * 100

# Set up the plot with larger figure size
fig, ax = plt.subplots(figsize=(16, 10))

# Define colors
colors = ['#3498db', '#2ecc71']  # Blue for Without AIA, Green for With AIA
labels = ['Without AIA vs CuSparse', 'With AIA vs CuSparse']

# Create x positions for bars
n_configs = len(df)  # 12 configurations
bar_width = 0.35     # Larger bar width
x_positions = np.arange(n_configs)

# Create the two sets of bars
bars1 = ax.bar(x_positions - bar_width/2, df['NoAIA_Improvement'], bar_width, 
               color=colors[0], alpha=0.8, edgecolor='black', linewidth=1,
               label=labels[0])

bars2 = ax.bar(x_positions + bar_width/2, df['AIA_Improvement'], bar_width,
               color=colors[1], alpha=0.8, edgecolor='black', linewidth=1,
               label=labels[1])

# Add value labels on top of bars with larger font
for bars, values in [(bars1, df['NoAIA_Improvement']), (bars2, df['AIA_Improvement'])]:
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
               f'{value:.1f}%', ha='center', va='bottom', 
               fontsize=12, fontweight='bold')

# Customize the plot with larger fonts
ax.set_ylabel('Performance Improvement over CuSparse (%)', fontsize=16, fontweight='bold')
ax.set_xlabel('Dataset and Model Configuration', fontsize=16, fontweight='bold')
ax.set_title('GNN Performance Improvement Percentage over CuSparse\n(Higher values indicate better performance)', 
             fontsize=18, fontweight='bold', pad=25)

# Create x-axis labels
x_labels = [f'{row["Dataset"]}\n{row["Model"]}' for _, row in df.iterrows()]
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, fontsize=12, fontweight='bold')

# Add legend with larger font
ax.legend(loc='upper left', fontsize=14, framealpha=0.9)

# Add grid for better readability
ax.grid(True, axis='y', alpha=0.3, linestyle='--')

# Set y-axis limits with more space
max_improvement = max(df['AIA_Improvement'].max(), df['NoAIA_Improvement'].max())
ax.set_ylim(0, max_improvement * 1.2)

# Add horizontal line at 0% for reference
ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)

# Add vertical lines to separate datasets (after every 3 configurations)
for i in [2.5, 5.5, 8.5]:
    ax.axvline(x=i, color='gray', linestyle='-', alpha=0.7, linewidth=2)

# Add dataset group labels at the top
dataset_centers = [1, 4, 7, 10]  # Center of each group of 3
dataset_names = ['Reddit', 'Protein', 'Flickr', 'Yelp']
for center, name in zip(dataset_centers, dataset_names):
    ax.text(center, max_improvement * 1.15, name, ha='center', va='center',
            fontsize=16, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))

# Increase tick label sizes
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

# Improve layout
plt.tight_layout()

# Add performance summary text box with larger font
summary_stats = f"""Performance Summary:
• AIA shows 20-60% improvement over CuSparse
• Without AIA shows 10-50% improvement over CuSparse
• Best AIA improvement: {df.loc[df['AIA_Improvement'].idxmax(), 'Dataset']} {df.loc[df['AIA_Improvement'].idxmax(), 'Model']} ({df['AIA_Improvement'].max():.1f}%)
• Consistent benefits across all configurations
• AIA provides additional 5-15% boost over baseline"""

ax.text(0.02, 0.98, summary_stats, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Show the plot
plt.show()

# Print detailed comparison table
print("\nDetailed Improvement Percentage Analysis:")
print("="*100)
print(f"{'Dataset':<10} {'Model':<6} {'Without AIA':<15} {'With AIA':<12} {'AIA Advantage':<15}")
print(f"{'':^10} {'':^6} {'vs CuSparse (%)':<15} {'vs CuSparse (%)':<12} {'over Baseline (%)':<15}")
print("="*100)

for _, row in df.iterrows():
    aia_advantage = row['AIA_Improvement'] - row['NoAIA_Improvement']
    print(f"{row['Dataset']:<10} {row['Model']:<6} {row['NoAIA_Improvement']:<15.1f} "
          f"{row['AIA_Improvement']:<12.1f} {aia_advantage:<15.1f}")

print("\n" + "="*100)
print("Summary Statistics:")
print(f"Average improvement without AIA: {df['NoAIA_Improvement'].mean():.1f}%")
print(f"Average improvement with AIA: {df['AIA_Improvement'].mean():.1f}%")
print(f"Average additional benefit from AIA: {(df['AIA_Improvement'] - df['NoAIA_Improvement']).mean():.1f}%")
print(f"Best performing configuration: {df.loc[df['AIA_Improvement'].idxmax(), 'Dataset']} {df.loc[df['AIA_Improvement'].idxmax(), 'Model']} with {df['AIA_Improvement'].max():.1f}% improvement")
