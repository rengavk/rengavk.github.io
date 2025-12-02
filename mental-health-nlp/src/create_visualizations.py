import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'data', 'mental_health_texts.csv')
df = pd.read_csv(data_path)

print(f"Loaded {len(df)} samples")

# Create output directory
output_dir = os.path.join(project_dir, 'visualizations')
os.makedirs(output_dir, exist_ok=True)

# Visualization 1: Class Distribution
plt.figure(figsize=(10, 6))
counts = df['label'].value_counts()
colors = ['#10b981', '#f59e0b', '#ef4444']  # Green, Orange, Red
plt.bar(counts.index, counts.values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
plt.title('Mental Health Text Classification Distribution', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Stress Level', fontsize=12, fontweight='bold')
plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
plt.xticks(['low_stress', 'high_stress', 'crisis'], 
           ['Low Stress\n(Normal)', 'High Stress\n(Elevated)', 'Crisis\n(Critical)'])
for i, v in enumerate(counts.values):
    plt.text(i, v + 20, str(v), ha='center', va='bottom', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
print("[OK] Class distribution saved")
plt.close()

# Visualization 2: Text Length Analysis
plt.figure(figsize=(12, 6))
for label, color in zip(['low_stress', 'high_stress', 'crisis'], colors):
    data = df[df['label'] == label]['text_length']
    plt.hist(data, bins=30, alpha=0.6, label=label.replace('_', ' ').title(), color=color, edgecolor='black')
plt.title('Text Length Distribution by Stress Level', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Text Length (characters)', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.legend(loc='upper right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'text_length_dist.png'), dpi=300, bbox_inches='tight')
print("[OK] Text length distribution saved")
plt.close()

# Visualization 3: Word Count Analysis
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
labels = ['low_stress', 'high_stress', 'crisis']
titles = ['Low Stress', 'High Stress', 'Crisis']

for ax, label, title, color in zip(axes, labels, titles, colors):
    data = df[df['label'] == label]['word_count']
    ax.hist(data, bins=20, color=color, alpha=0.7, edgecolor='black')
    ax.set_title(f'{title}\n(Avg: {data.mean():.1f} words)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Word Count', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.grid(alpha=0.3)

plt.suptitle('Word Count Distribution by Stress Level', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'word_count_analysis.png'), dpi=300, bbox_inches='tight')
print("[OK] Word count analysis saved")
plt.close()

# Visualization 4: Summary Statistics
fig, ax = plt.subplots(figsize=(10, 6))

summary_data = df.groupby('label').agg({
    'text_length': 'mean',
    'word_count': 'mean'
}).round(1)

x = range(len(summary_data))
width = 0.35

bars1 = ax.bar([i - width/2 for i in x], summary_data['text_length'], width, 
               label='Avg Text Length', color='#3b82f6', alpha=0.8, edgecolor='black')
bars2 = ax.bar([i + width/2 for i in x], summary_data['word_count'] * 6, width,
               label='Avg Word Count (×6)', color='#8b5cf6', alpha=0.8, edgecolor='black')

ax.set_xlabel('Stress Level', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Average Statistics by Stress Level', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([l.replace('_', ' ').title() for l in summary_data.index])
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'summary_stats.png'), dpi=300, bbox_inches='tight')
print("[OK] Summary statistics saved")
plt.close()

print("\n" + "="*60)
print("All visualizations created successfully!")
print("="*60)
print(f"Output directory: {output_dir}")
