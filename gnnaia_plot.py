import matplotlib.pyplot as plt
import numpy as np

# Data for SpGEMM AIA improvement over without AIA
datasets = ['Reddit', 'Flickr', 'Yelp', 'Protein']
improvements = [23.555, 32.496, 26.543, 17.0876]

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Create bars
bars = ax.bar(datasets, improvements, 
              color=['#3498db', '#e74c3c', '#f39c12', '#9b59b6'], 
              alpha=0.8, 
              edgecolor='black', 
              linewidth=1.5,
              width=0.6)

# Add value labels on top of bars
for bar, value in zip(bars, improvements):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
           f'{value:.3f}%', 
           ha='center', va='bottom', 
           fontsize=12, fontweight='bold')

# Customize the plot
ax.set_ylabel('Improvement Percentage (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax.set_title('SpGEMM AIA Improvement Over Without AIA\nAcross Different Datasets', 
             fontsize=14, fontweight='bold', pad=20)

# Add grid for better readability
ax.grid(True, axis='y', alpha=0.3, linestyle='--')

# Set y-axis to start from 0 and add some padding at the top
ax.set_ylim(0, max(improvements) * 1.15)

# Customize x-axis
ax.tick_params(axis='x', labelsize=11)
ax.tick_params(axis='y', labelsize=10)

# Add a horizontal line at the average for reference
avg_improvement = np.mean(improvements)
ax.axhline(y=avg_improvement, color='red', linestyle=':', alpha=0.7, linewidth=2)
ax.text(len(datasets) - 0.5, avg_improvement + 1, f'Average: {avg_improvement:.2f}%', 
        ha='right', va='bottom', fontsize=10, color='red', fontweight='bold')

# Add summary statistics text box
summary_text = f"""Summary Statistics:
• Best improvement: {datasets[np.argmax(improvements)]} ({max(improvements):.3f}%)
• Lowest improvement: {datasets[np.argmin(improvements)]} ({min(improvements):.3f}%)
• Average improvement: {avg_improvement:.2f}%
• Standard deviation: {np.std(improvements):.2f}%"""

ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()

# Print the data table
print("\nSpGEMM AIA Improvement Data:")
print("="*40)
print(f"{'Dataset':<10} {'Improvement (%)':<15}")
print("="*40)
for dataset, improvement in zip(datasets, improvements):
    print(f"{dataset:<10} {improvement:<15.3f}")
print("="*40)
print(f"{'Average':<10} {avg_improvement:<15.2f}")
print(f"{'Std Dev':<10} {np.std(improvements):<15.2f}")
print(f"{'Range':<10} {max(improvements) - min(improvements):<15.3f}")
