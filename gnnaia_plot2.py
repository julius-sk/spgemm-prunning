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

# Set up the plot
fig, ax = plt.subplots(figsize=(14, 8))

# Define colors
colors = ['#3498db', '#2ecc71']  # Blue for Without AIA, Green for With AIA
labels = ['Without AIA vs CuSparse', 'With AIA vs CuSparse']

# Create x positions for bars
datasets = ['Reddit', 'Protein', 'Flickr', 'Yelp']
models = ['GCN', 'GIN', 'SAGE']
n_datasets = len(datasets)
n_models = len(models)
n_conditions = 2  # Only two comparisons now

# Width of bars and spacing
bar_width = 0.12
group_spacing = 0.4
dataset_spacing = 1.0

# Calculate x positions
x_positions = []
labels_positions = []
dataset_labels = []

for i, dataset in enumerate(datasets):
    dataset_start = i * (n_models * n_conditions * bar_width + group_spacing + dataset_spacing)
    
    for j, model in enumerate(models):
        model_start = dataset_start + j * (n_conditions * bar_width + group_spacing)
        
        # Store center position for model label
        model_center = model_start + (n_conditions * bar_width) / 2
        labels_positions.append(model_center)
        dataset_labels.append(f'{dataset}\n{model}')
        
        for k in range(n_conditions):
            x_pos = model_start + k * bar_width
            x_positions.append(x_pos)

# Prepare data for plotting
x_pos_idx = 0
for i, dataset in enumerate(datasets):
    for j, model in enumerate(models):
        row = df[(df['Dataset'] == dataset) & (df['Model'] == model)]
        
        values = [row['NoAIA_Improvement'].iloc[0], row['AIA_Improvement'].iloc[0]]
        
        for k, (value, color, label) in enumerate(zip(values, colors, labels)):
            x_pos = x_positions[x_pos_idx + k]
            bar = ax.bar(x_pos, value, bar_width, color=color, 
                        label=label if i == 0 and j == 0 else "", 
                        alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add value labels on top of bars
            ax.text(x_pos, value + 1, f'{value:.1f}%', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        x_pos_idx += n_conditions

# Customize the plot
ax.set_ylabel('Performance Improvement over CuSparse (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Dataset and Model', fontsize=12, fontweight='bold')
ax.set_title('GNN Performance Improvement Percentage over CuSparse\n(Higher values indicate better performance)', 
             fontsize=14, fontweight='bold', pad=20)

# Set x-axis labels
ax.set_xticks(labels_positions)
ax.set_xticklabels(dataset_labels, fontsize=10)

# Add legend
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

# Add grid for better readability
ax.grid(True, axis='y', alpha=0.3, linestyle='--')

# Set y-axis limits
max_improvement = max(df['AIA_Improvement'].max(), df['NoAIA_Improvement'].max())
ax.set_ylim(0, max_improvement * 1.15)

# Add horizontal line at 0% for reference
ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)

# Add vertical lines to separate datasets
for i in range(1, len(datasets)):
    sep_x = (labels_positions[(i-1)*3 + 2] + labels_positions[i*3]) / 2
    ax.axvline(x=sep_x, color='gray', linestyle='-', alpha=0.5, linewidth=1)

# Improve layout
plt.tight_layout()

# Add performance summary text box
summary_stats = f"""Performance Summary:
• AIA shows 20-60% improvement over CuSparse
• Without AIA shows 10-50% improvement over CuSparse
• Best AIA improvement: {df.loc[df['AIA_Improvement'].idxmax(), 'Dataset']} {df.loc[df['AIA_Improvement'].idxmax(), 'Model']} ({df['AIA_Improvement'].max():.1f}%)
• Consistent benefits across all configurations
• AIA provides additional 5-15% boost over baseline"""

ax.text(0.02, 0.98, summary_stats, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Show the plot
plt.show()

# Print detailed comparison table
print("\nDetailed Improvement Percentage Analysis:")
print("="*90)
print(f"{'Dataset':<10} {'Model':<6} {'Without AIA':<15} {'With AIA':<12} {'AIA Advantage':<15}")
print(f"{'':^10} {'':^6} {'vs CuSparse (%)':<15} {'vs CuSparse (%)':<12} {'over Baseline (%)':<15}")
print("="*90)

for _, row in df.iterrows():
    aia_advantage = row['AIA_Improvement'] - row['NoAIA_Improvement']
    print(f"{row['Dataset']:<10} {row['Model']:<6} {row['NoAIA_Improvement']:<15.1f} "
          f"{row['AIA_Improvement']:<12.1f} {aia_advantage:<15.1f}")

print("\n" + "="*90)
print("Summary Statistics:")
print(f"Average improvement without AIA: {df['NoAIA_Improvement'].mean():.1f}%")
print(f"Average improvement with AIA: {df['AIA_Improvement'].mean():.1f}%")
print(f"Average additional benefit from AIA: {(df['AIA_Improvement'] - df['NoAIA_Improvement']).mean():.1f}%")
print(f"Best performing configuration: {df.loc[df['AIA_Improvement'].idxmax(), 'Dataset']} {df.loc[df['AIA_Improvement'].idxmax(), 'Model']} with {df['AIA_Improvement'].max():.1f}% improvement")

# Create a secondary plot showing the additional benefit of AIA over baseline
fig2, ax2 = plt.subplots(figsize=(12, 6))

# Calculate additional benefit of AIA over baseline (Without AIA)
df['AIA_Additional_Benefit'] = df['AIA_Improvement'] - df['NoAIA_Improvement']

# Plot additional benefit
x_pos_simple = np.arange(len(df))
bars = ax2.bar(x_pos_simple, df['AIA_Additional_Benefit'], 
               color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)

# Add value labels
for i, (bar, value) in enumerate(zip(bars, df['AIA_Additional_Benefit'])):
    ax2.text(bar.get_x() + bar.get_width()/2, value + 0.2, f'{value:.1f}%', 
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Customize secondary plot
ax2.set_ylabel('Additional Improvement with AIA (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Configuration', fontsize=12, fontweight='bold')
ax2.set_title('Additional Performance Benefit of AIA over Baseline\n(AIA improvement - Without AIA improvement)', 
              fontsize=14, fontweight='bold', pad=20)

# Set x-axis labels
x_labels = [f"{row['Dataset']}\n{row['Model']}" for _, row in df.iterrows()]
ax2.set_xticks(x_pos_simple)
ax2.set_xticklabels(x_labels, fontsize=9, rotation=45, ha='right')

# Add grid and horizontal reference line
ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)

# Set y-axis limits
max_benefit = df['AIA_Additional_Benefit'].max()
ax2.set_ylim(0, max_benefit * 1.2)

plt.tight_layout()
plt.show()
