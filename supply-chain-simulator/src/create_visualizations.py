import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create output directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_dir = os.path.join(project_dir, 'visualizations')
os.makedirs(output_dir, exist_ok=True)

# Simulate supply chain data
np.random.seed(42)
days = 365

# Generate stockout data (before and after optimization)
before_stockouts = np.random.poisson(20, days)  # Average 20 stockouts per day
after_stockouts = np.random.poisson(14, days)   # 30% reduction

# Generate inventory levels for 4 echelons
factory_inventory = 500 + np.random.normal(0, 50, days).cumsum()
warehouse_inventory = 300 + np.random.normal(0, 30, days).cumsum()
distributor_inventory = 200 + np.random.normal(0, 20, days).cumsum()
retailer_inventory = 100 + np.random.normal(0, 15, days).cumsum()

# Visualization 1: Stockout Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Before/After comparison
labels = ['Before\nOptimization', 'After\nOptimization']
values = [before_stockouts.sum(), after_stockouts.sum()]
colors = ['#ef4444', '#10b981']
bars = ax1.bar(labels, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Total Stockouts (365 days)', fontsize=12, fontweight='bold')
ax1.set_title('Stockout Reduction Impact', fontsize=14, fontweight='bold', pad=20)

# Add value labels and reduction percentage
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(val)}\nstockouts', ha='center', va='bottom', fontweight='bold', fontsize=11)

reduction = ((values[0] - values[1]) / values[0]) * 100
ax1.text(0.5, max(values) * 0.5, f'{reduction:.1f}%\nReduction',
         ha='center', fontsize=16, fontweight='bold', color='green',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', linewidth=2))

# Service level improvement
service_before = 85
service_after = 95
ax2.bar(['Before', 'After'], [service_before, service_after],
        color=['#f59e0b', '#10b981'], alpha=0.7, edgecolor='black', linewidth=2)
ax2.axhline(y=90, color='red', linestyle='--', label='Target (90%)', linewidth=2)
ax2.set_ylabel('Service Level (%)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_title('Service Level Improvement', fontsize=14, fontweight='bold', pad=20)
ax2.set_ylim([0, 100])

for i, (label, val) in enumerate(zip(['Before', 'After'], [service_before, service_after])):
    ax2.text(i, val + 2, f'{val}%', ha='center', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'stockout_analysis.png'), dpi=300, bbox_inches='tight')
print("[OK] Stockout analysis saved")
plt.close()

# Visualization 2: Multi-Echelon Inventory Levels
plt.figure(figsize=(14, 7))
days_range = range(90)  # Show 90 days

plt.plot(days_range, factory_inventory[:90], label='Factory', linewidth=2.5, color='#3b82f6')
plt.plot(days_range, warehouse_inventory[:90], label='Warehouse', linewidth=2.5, color='#8b5cf6')
plt.plot(days_range, distributor_inventory[:90], label='Distributor', linewidth=2.5, color='#ec4899')
plt.plot(days_range, retailer_inventory[:90], label='Retailer', linewidth=2.5, color='#f59e0b')

plt.xlabel('Days', fontsize=12, fontweight='bold')
plt.ylabel('Inventory Units', fontsize=12, fontweight='bold')
plt.title('Multi-Echelon Inventory Levels Over Time', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'inventory_levels.png'), dpi=300, bbox_inches='tight')
print("[OK] Inventory levels saved")
plt.close()

# Visualization 3: Cost Breakdown
costs = {
    'Holding\nCost': 50000,
    'Ordering\nCost': 15000,
    'Backorder\nPenalty': 8000,
    'Transportation': 12000
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Pie chart
colors_pie = ['#3b82f6', '#10b981', '#ef4444', '#f59e0b']
wedges, texts, autotexts = ax1.pie(costs.values(), labels=costs.keys(), autopct='%1.1f%%',
                                     colors=colors_pie, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax1.set_title('Cost Distribution', fontsize=14, fontweight='bold', pad=20)

# Bar chart with before/after
categories = list(costs.keys())
before_costs = list(costs.values())
after_costs = [c * 0.85 for c in before_costs]  # 15% reduction

x = np.arange(len(categories))
width = 0.35

bars1 = ax2.bar(x - width/2, before_costs, width, label='Before', color='#ef4444', alpha=0.7, edgecolor='black')
bars2 = ax2.bar(x + width/2, after_costs, width, label='After', color='#10b981', alpha=0.7, edgecolor='black')

ax2.set_ylabel('Cost ($)', fontsize=12, fontweight='bold')
ax2.set_title('Cost Reduction After Optimization', fontsize=14, fontweight='bold', pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels(categories, fontsize=10)
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cost_analysis.png'), dpi=300, bbox_inches='tight')
print("[OK] Cost analysis saved")
plt.close()

# Visualization 4: Lead Time Distribution
plt.figure(figsize=(12, 6))

echelons = ['Factory→\nWarehouse', 'Warehouse→\nDistributor', 'Distributor→\nRetailer']
lead_times_min = [7, 3, 1]
lead_times_max = [10, 5, 2]
lead_times_avg = [(min_val + max_val) / 2 for min_val, max_val in zip(lead_times_min, lead_times_max)]

x_pos = np.arange(len(echelons))
colors_lead = ['#3b82f6', '#8b5cf6', '#ec4899']

bars = plt.bar(x_pos, lead_times_avg, color=colors_lead, alpha=0.7, edgecolor='black', linewidth=2)

# Calculate error values properly
yerr_lower = [avg - min_val for avg, min_val in zip(lead_times_avg, lead_times_min)]
yerr_upper = [max_val - avg for avg, max_val in zip(lead_times_avg, lead_times_max)]

plt.errorbar(x_pos, lead_times_avg, 
             yerr=[yerr_lower, yerr_upper],
             fmt='none', ecolor='black', capsize=10, capthick=2)

plt.xlabel('Supply Chain Echelon', fontsize=12, fontweight='bold')
plt.ylabel('Lead Time (days)', fontsize=12, fontweight='bold')
plt.title('Lead Time Distribution Across Echelons', fontsize=16, fontweight='bold', pad=20)
plt.xticks(x_pos, echelons, fontsize=11)
plt.grid(alpha=0.3, axis='y')

for bar, avg in zip(bars, lead_times_avg):
    plt.text(bar.get_x() + bar.get_width()/2., avg + 0.3,
             f'{avg:.1f} days', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'lead_time_analysis.png'), dpi=300, bbox_inches='tight')
print("[OK] Lead time analysis saved")
plt.close()

print("\n" + "="*60)
print("All visualizations created successfully!")
print("="*60)
print(f"Output directory: {output_dir}")
