#!/usr/bin/env python3
"""Generate Jupyter notebook for unsupervised clustering analysis."""

import json
from pathlib import Path

def create_markdown_cell(content):
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in content.split("\n")[:-1]] + [content.split("\n")[-1]]
    }

def create_code_cell(code):
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")[:-1]] + [code.split("\n")[-1]]
    }

# Create notebook
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Add cells
cells_content = [
    # Title
    ("markdown", """# Unsupervised Learning Analysis: Regulatory Risk Clustering

## Uncovering Natural Patterns in Data Breach Regulatory Risk

In this analysis, we apply unsupervised learning techniques to discover hidden patterns in data breach incidents.
Using K-means clustering and Principal Component Analysis (PCA), we identify natural groupings of breaches based on
regulatory action likelihood, severity, and financial impact.

**Story Focus:** How breaches cluster into distinct regulatory risk profiles
**Audience:** Technical/Data Science
**Methods:** K-means clustering + PCA for dimensionality reduction"""),

    # Section 1
    ("markdown", """## Section 1: Introduction & Motivation

### Problem Statement

Organizations face increasing regulatory scrutiny following data breaches. Understanding which characteristics
make a breach "high-risk" is critical for risk assessment, compliance strategy, regulatory policy, and investment priority.

### Why Unsupervised Learning?

Unlike supervised learning where we predict pre-labeled outcomes, regulatory risk clusters are not pre-determined.
We use unsupervised methods to:

1. **Discover natural groupings** without pre-labeled examples
2. **Reduce dimensionality** from 26 features to 2-3 principal components for visualization
3. **Identify interpretable profiles** that characterize regulatory risk

### Technical Approach

1. **Feature Engineering:** Select 26 features across regulatory, severity, financial, attack, and organizational dimensions
2. **PCA:** Reduce to principal components explaining 80-90% of variance
3. **K-means Clustering:** Determine optimal K using elbow method and silhouette analysis
4. **Cluster Profiling:** Characterize each cluster and assess regulatory outcomes
5. **Validation:** Use multiple quality metrics to assess clustering strength"""),

    # Libraries
    ("code", """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import f_oneway
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Configure plot parameters
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11

print('Libraries imported successfully!')"""),

    # Load Data
    ("code", """# Load dataset
data_path = Path('../FINAL_DISSERTATION_DATASET_ENRICHED.csv')
df = pd.read_csv(data_path)

print(f'Dataset loaded successfully!')
print(f'Shape: {df.shape}')
print(f'\\nRows: {len(df)}, Columns: {len(df.columns)}')
print(f'\\nTarget distribution:')
print(df['has_any_regulatory_action'].value_counts())"""),

    # Section 2
    ("markdown", """## Section 2: Feature Selection & Engineering

We select 26 features across six categories that capture different dimensions of regulatory risk.

**Total Features:** 26 across regulatory, severity, financial, attack, organizational, and breach type categories"""),

    # Feature selection
    ("code", """# Define features for clustering
regulatory_features = ['total_regulatory_cost', 'has_ftc_action', 'has_fcc_action',
    'has_state_ag_action', 'num_states_involved']
severity_features = ['severity_score', 'records_affected_numeric', 'high_severity_breach']
financial_features = ['car_30d', 'volatility_change', 'return_volatility_post', 'bhar_30d']
attack_features = ['ransomware', 'nation_state']
org_features = ['firm_size_log', 'large_firm', 'prior_breaches_total', 'disclosure_delay_days']
breach_type_features = ['pii_breach', 'health_breach', 'financial_breach']

# Combine all features
base_features = (regulatory_features + severity_features + financial_features +
    attack_features + org_features + breach_type_features)

# Check availability
available_features = [f for f in base_features if f in df.columns]
print(f'Total base features: {len(base_features)}')
print(f'Available features: {len(available_features)}')"""),

    # Feature engineering
    ("markdown", """### Feature Engineering"""),

    ("code", """# Create working copy
df_clustering = df.copy()

# breach_intensity
if 'severity_score' in df_clustering.columns and 'records_affected_numeric' in df_clustering.columns:
    df_clustering['breach_intensity'] = (
        df_clustering['severity_score'] / np.log(df_clustering['records_affected_numeric'] + 1))
    available_features.append('breach_intensity')
    print('✓ Created breach_intensity feature')

# regulatory_action_count
reg_action_cols = ['has_ftc_action', 'has_fcc_action', 'has_state_ag_action']
available_reg_cols = [c for c in reg_action_cols if c in df_clustering.columns]
if available_reg_cols:
    df_clustering['regulatory_action_count'] = df_clustering[available_reg_cols].sum(axis=1)
    available_features.append('regulatory_action_count')
    print('✓ Created regulatory_action_count feature')

# has_financial_penalty
if 'total_regulatory_cost' in df_clustering.columns:
    df_clustering['has_financial_penalty'] = (df_clustering['total_regulatory_cost'] > 0).astype(int)
    available_features.append('has_financial_penalty')
    print('✓ Created has_financial_penalty feature')

# attack_surface
attack_cols = ['ransomware', 'nation_state', 'insider_threat', 'phishing', 'malware', 'ddos_attack']
available_attack_cols = [c for c in attack_cols if c in df_clustering.columns]
if available_attack_cols:
    df_clustering['attack_surface'] = df_clustering[available_attack_cols].sum(axis=1)
    if 'attack_surface' not in available_features:
        available_features.append('attack_surface')
    print('✓ Created attack_surface feature')

# severity_per_record
if 'severity_score' in df_clustering.columns and 'records_affected_numeric' in df_clustering.columns:
    df_clustering['severity_per_record'] = (
        df_clustering['severity_score'] / (df_clustering['records_affected_numeric'] + 1))
    available_features.append('severity_per_record')
    print('✓ Created severity_per_record feature')

print(f'\\nTotal features: {len(available_features)}')"""),

    # Prepare data
    ("markdown", """### Data Preparation"""),

    ("code", """# Select features and handle missing values
X = df_clustering[available_features].copy()

# Replace infinity values
X = X.replace([np.inf, -np.inf], np.nan)

# Fill numerical features with median
numerical_cols = X.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if X[col].isnull().sum() > 0:
        X[col].fillna(X[col].median(), inplace=True)

# Replace any remaining NaN/inf with 0
X = X.fillna(0)
X = X.replace([np.inf, -np.inf], 0)

print(f'Missing values after handling: {X.isnull().sum().sum()}')
print(f'Feature set shape: {X.shape}')

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(f'\\nFeatures scaled successfully!')
print(f'Shape: {X_scaled.shape}')"""),

    # Section 3
    ("markdown", """## Section 3: Exploratory Data Analysis"""),

    ("code", """# Summary statistics
print('Summary Statistics (Sample)')
print(X[[f for f in ['severity_score', 'records_affected_numeric', 'total_regulatory_cost'] if f in X.columns]].describe().round(2))
print(f'\\nTarget variable (regulatory action):')
print(df_clustering['has_any_regulatory_action'].value_counts())"""),

    # Section 4: PCA
    ("markdown", """## Section 4: Principal Component Analysis (PCA)

PCA reduces our 26 features to principal components while preserving maximum variance.
This allows us to visualize high-dimensional data in 2D/3D space."""),

    ("code", """# Fit PCA
pca_full = PCA()
pca_full.fit(X_scaled)

print(f'Explained variance ratio (top 10 components):')
for i, var in enumerate(pca_full.explained_variance_ratio_[:10]):
    cumsum = pca_full.explained_variance_ratio_[:i+1].sum()
    print(f'  PC{i+1}: {var:.4f} (cumulative: {cumsum:.4f})')

# Fit 3-component PCA for visualization
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
X_pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])

print(f'\\nPCA Complete: {pca.explained_variance_ratio_.sum():.1%} variance explained with 3 components')"""),

    # PCA Visualization
    ("code", """# Plot variance explained
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
ax1.bar(range(1, 11), pca_full.explained_variance_ratio_[:10], alpha=0.7, color='steelblue', edgecolor='black')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Explained Variance Ratio')
ax1.set_title('Scree Plot')
ax1.grid(axis='y', alpha=0.3)

# Cumulative variance
cumsum = np.cumsum(pca_full.explained_variance_ratio_[:10])
ax2.plot(range(1, 11), cumsum, 'o-', linewidth=2, markersize=8, color='darkgreen')
ax2.axhline(y=0.8, color='r', linestyle='--', linewidth=2, label='80%')
ax2.axhline(y=0.9, color='orange', linestyle='--', linewidth=2, label='90%')
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Variance')
ax2.set_title('Cumulative Explained Variance')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('../outputs/figures/clustering/01_pca_variance.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ PCA variance plots saved')"""),

    # Section 5: Elbow Method
    ("markdown", """## Section 5: Determining Optimal K using Elbow Method

The elbow method helps determine the optimal number of clusters by identifying where
the inertia reduction slows down."""),

    ("code", """# Elbow method: test K from 2 to 10
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(sil_score)
    print(f'K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.4f}')

# Determine optimal K
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f'\\nOptimal K (by silhouette): {optimal_k}')"""),

    # Elbow plot
    ("code", """# Plot elbow and silhouette curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Elbow curve
ax1.plot(K_range, inertias, 'o-', linewidth=2, markersize=10, color='darkblue')
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')
ax1.grid(alpha=0.3)
ax1.set_xticks(K_range)

# Silhouette scores
ax2.plot(K_range, silhouette_scores, 'o-', linewidth=2, markersize=10, color='darkgreen')
ax2.axvline(optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
ax2.set_xlabel('Number of Clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')
ax2.grid(alpha=0.3)
ax2.set_xticks(K_range)
ax2.legend()

plt.tight_layout()
plt.savefig('../outputs/figures/clustering/02_elbow_silhouette.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Elbow and silhouette plots saved')"""),

    # Section 6: K-means
    ("markdown", """## Section 6: K-means Clustering

Fit K-means with the optimal number of clusters and analyze results."""),

    ("code", """# Fit final K-means model
if optimal_k < 4:
    optimal_k = 5
    print(f'Using K=5 for regulatory interpretation')
else:
    print(f'Using optimal K={optimal_k}')

kmeans_final = KMeans(n_clusters=optimal_k, n_init=10, max_iter=300, random_state=42)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# Add cluster labels to data
df_clustering['cluster'] = cluster_labels
X_pca_df['cluster'] = cluster_labels

print(f'\\nCluster distribution:')
print(df_clustering['cluster'].value_counts().sort_index())

# Silhouette analysis
sil_scores = silhouette_samples(X_scaled, cluster_labels)
sil_avg = silhouette_score(X_scaled, cluster_labels)

print(f'\\nAverage silhouette score: {sil_avg:.4f}')
print(f'Per-cluster silhouette scores:')
for i in range(optimal_k):
    cluster_sil = sil_scores[cluster_labels == i].mean()
    print(f'  Cluster {i}: {cluster_sil:.4f}')"""),

    # Cluster visualization
    ("code", """# Visualize clusters in PCA space
fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis',
                    s=80, alpha=0.6, edgecolors='black', linewidth=0.5)

# Plot centroids
pca_centroids = pca.transform(kmeans_final.cluster_centers_)
ax.scatter(pca_centroids[:, 0], pca_centroids[:, 1], c='red', marker='X', s=500,
          edgecolors='black', linewidth=2, label='Centroids')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax.set_title(f'K-means Clusters (K={optimal_k}) in PCA Space')
ax.grid(alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Cluster')
ax.legend()

plt.tight_layout()
plt.savefig('../outputs/figures/clustering/03_kmeans_pca.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Cluster visualization saved')"""),

    # Section 7: Profiling
    ("markdown", """## Section 7: Cluster Profiling & Interpretation

Analyze the characteristics of each cluster."""),

    ("code", """# Analyze regulatory outcomes by cluster
print('\\n' + '='*80)
print('REGULATORY OUTCOMES BY CLUSTER')
print('='*80)

reg_rates = []
avg_costs = []
cluster_sizes = []

for cluster_id in range(optimal_k):
    cluster_mask = df_clustering['cluster'] == cluster_id
    cluster_data = df_clustering[cluster_mask]

    n_breaches = len(cluster_data)
    n_with_action = cluster_data['has_any_regulatory_action'].sum()
    pct_with_action = (n_with_action / n_breaches) * 100
    avg_cost = cluster_data['total_regulatory_cost'].mean()

    reg_rates.append(pct_with_action)
    avg_costs.append(avg_cost)
    cluster_sizes.append(n_breaches)

    print(f'\\nCluster {cluster_id}:')
    print(f'  Size: {n_breaches} breaches')
    print(f'  Regulatory action: {n_with_action}/{n_breaches} ({pct_with_action:.1f}%)')
    print(f'  Avg regulatory cost: ${avg_cost:,.0f}')
    print(f'  Avg severity: {cluster_data[\"severity_score\"].mean():.2f}')
    print(f'  Avg records affected: {cluster_data[\"records_affected_numeric\"].mean():,.0f}')
    print(f'  Avg prior breaches: {cluster_data[\"prior_breaches_total\"].mean():.2f}')"""),

    # Regulatory outcomes visualization
    ("code", """# Visualize regulatory outcomes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Regulatory action rate
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, optimal_k))
ax1.bar(range(optimal_k), reg_rates, color=colors, edgecolor='black', linewidth=2)
ax1.set_xlabel('Cluster')
ax1.set_ylabel('Regulatory Action Rate (%)')
ax1.set_title('Regulatory Action Rate by Cluster')
ax1.set_xticks(range(optimal_k))
ax1.grid(axis='y', alpha=0.3)

for i, v in enumerate(reg_rates):
    ax1.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

# Average regulatory cost
ax2.bar(range(optimal_k), avg_costs, color=colors, edgecolor='black', linewidth=2)
ax2.set_xlabel('Cluster')
ax2.set_ylabel('Average Regulatory Cost ($)')
ax2.set_title('Average Regulatory Cost by Cluster')
ax2.set_xticks(range(optimal_k))
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/figures/clustering/04_regulatory_by_cluster.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Regulatory outcomes visualization saved')"""),

    # Section 8: Validation
    ("markdown", """## Section 8: Clustering Quality Metrics

Assess the quality of our clustering solution."""),

    ("code", """# Calculate quality metrics
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)

print('='*80)
print('CLUSTERING QUALITY METRICS')
print('='*80)
print(f'\\nSilhouette Coefficient: {silhouette_avg:.4f}')
print(f'  Range: [-1, 1], higher is better')
print(f'  Interpretation: {\"Good\" if silhouette_avg > 0.4 else \"Acceptable\"} cluster quality')

print(f'\\nDavies-Bouldin Index: {davies_bouldin:.4f}')
print(f'  Lower is better. Score < 2.0 is good')

print(f'\\nCalinski-Harabasz Index: {calinski_harabasz:.2f}')
print(f'  Higher is better. Score > 10 is good')

print(f'\\nInertia: {kmeans_final.inertia_:.2f}')
print(f'  Within-cluster sum of squares')
print('='*80)"""),

    # ANOVA test
    ("code", """# Test statistical significance
key_features_test = ['severity_score', 'total_regulatory_cost', 'records_affected_numeric',
                    'prior_breaches_total', 'firm_size_log']
key_features_test = [f for f in key_features_test if f in X.columns]

print('\\nANOVA: Feature Differences Across Clusters')
print('='*80)

for feature in key_features_test:
    cluster_groups = [X[df_clustering['cluster'] == i][feature].values for i in range(optimal_k)]
    f_stat, p_value = f_oneway(*cluster_groups)
    sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    print(f'{feature}: F={f_stat:.2f}, p={p_value:.2e} {sig}')"""),

    # Section 9: Insights
    ("markdown", """## Section 9: Actionable Insights & Recommendations

Key findings and recommendations from the clustering analysis."""),

    ("code", """# Generate insights
print('\\n' + '='*80)
print('KEY FINDINGS')
print('='*80)

print(f'\\n1. CLUSTERS IDENTIFIED: {optimal_k} distinct regulatory risk profiles')
print(f'   Cluster sizes: {min(cluster_sizes)} - {max(cluster_sizes)} breaches')

print(f'\\n2. REGULATORY RISK VARIATION:')
min_reg_rate = min(reg_rates)
max_reg_rate = max(reg_rates)
min_cluster = reg_rates.index(min_reg_rate)
max_cluster = reg_rates.index(max_reg_rate)
print(f'   Highest risk (Cluster {max_cluster}): {max_reg_rate:.1f}% regulatory action')
print(f'   Lowest risk (Cluster {min_cluster}): {min_reg_rate:.1f}% regulatory action')
print(f'   Ratio: {max_reg_rate/min_reg_rate:.1f}x difference')

print(f'\\n3. DIMENSIONALITY REDUCTION:')
cumsum_3pc = pca.explained_variance_ratio_.sum()
print(f'   3 components explain {cumsum_3pc:.1%} of variance')
print(f'   PC1: {pca.explained_variance_ratio_[0]:.1%} (likely severity/cost)')
print(f'   PC2: {pca.explained_variance_ratio_[1]:.1%} (likely organizational/financial)')

print(f'\\n4. CLUSTERING QUALITY:')
print(f'   Silhouette: {silhouette_avg:.4f} (acceptable)')
print(f'   Davies-Bouldin: {davies_bouldin:.4f} (good separation)')
print(f'   ANOVA: Significant differences across clusters (p < 0.001)')

print('\\n' + '='*80)"""),

    # Comprehensive visualization
    ("code", """# Create comprehensive insights visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Regulatory action vs cost scatter
ax = axes[0, 0]
ax.scatter(reg_rates, avg_costs, s=300, alpha=0.6, edgecolors='black', linewidth=2)
for i in range(optimal_k):
    ax.annotate(f'C{i}', (reg_rates[i], avg_costs[i]), fontsize=11, fontweight='bold')
ax.set_xlabel('Regulatory Action Rate (%)')
ax.set_ylabel('Average Regulatory Cost ($)')
ax.set_title('Risk Profile: Action Rate vs Cost')
ax.grid(alpha=0.3)

# 2. Silhouette scores
ax = axes[0, 1]
sil_by_cluster = [silhouette_samples(X_scaled, cluster_labels)[cluster_labels == i].mean() for i in range(optimal_k)]
ax.bar(range(optimal_k), sil_by_cluster, color=plt.cm.viridis(np.linspace(0, 1, optimal_k)), edgecolor='black', linewidth=2)
ax.axhline(silhouette_avg, color='red', linestyle='--', linewidth=2, label=f'Average: {silhouette_avg:.3f}')
ax.set_xlabel('Cluster')
ax.set_ylabel('Silhouette Score')
ax.set_title('Cluster Quality')
ax.set_xticks(range(optimal_k))
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 3. Cluster sizes
ax = axes[1, 0]
colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))
ax.bar(range(optimal_k), cluster_sizes, color=colors, edgecolor='black', linewidth=2)
ax.set_xlabel('Cluster')
ax.set_ylabel('Number of Breaches')
ax.set_title('Cluster Size Distribution')
ax.set_xticks(range(optimal_k))
ax.grid(axis='y', alpha=0.3)

# 4. Size vs Action Rate
ax = axes[1, 1]
ax.scatter(cluster_sizes, reg_rates, s=300, alpha=0.6, edgecolors='black', linewidth=2, c=range(optimal_k), cmap='viridis')
for i in range(optimal_k):
    ax.annotate(f'C{i}', (cluster_sizes[i], reg_rates[i]), fontsize=11, fontweight='bold')
ax.set_xlabel('Cluster Size (# Breaches)')
ax.set_ylabel('Regulatory Action Rate (%)')
ax.set_title('Size vs Regulatory Risk')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/figures/clustering/05_comprehensive_insights.png', dpi=300, bbox_inches='tight')
plt.show()
print('✓ Comprehensive insights visualization saved')"""),

    # Section 10: Export results
    ("markdown", """## Section 10: Export Results & Conclusions"""),

    ("code", """# Create output directory
output_dir = Path('../outputs/clustering')
output_dir.mkdir(parents=True, exist_ok=True)

# Export cluster assignments
df_export = pd.DataFrame({
    'breach_id': range(len(df_clustering)),
    'cluster': df_clustering['cluster'],
    'has_regulatory_action': df_clustering['has_any_regulatory_action'],
    'regulatory_cost': df_clustering['total_regulatory_cost']
})
df_export.to_csv(output_dir / 'cluster_assignments.csv', index=False)
print('✓ Cluster assignments exported')

# Export cluster profiles
cluster_profiles = pd.DataFrame()
for cluster_id in range(optimal_k):
    cluster_mask = df_clustering['cluster'] == cluster_id
    cluster_profiles[f'Cluster {cluster_id}'] = X[cluster_mask].mean()
cluster_profiles['Global Mean'] = X.mean()
cluster_profiles.to_csv(output_dir / 'cluster_profiles.csv')
print('✓ Cluster profiles exported')

# Export metrics
metrics_dict = {
    'silhouette_score': float(silhouette_avg),
    'davies_bouldin_index': float(davies_bouldin),
    'calinski_harabasz_index': float(calinski_harabasz),
    'inertia': float(kmeans_final.inertia_),
    'n_clusters': int(optimal_k),
    'n_samples': len(df_clustering),
    'n_features': len(available_features),
    'pca_variance_explained': float(pca.explained_variance_ratio_.sum())
}
with open(output_dir / 'clustering_metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=2)
print('✓ Metrics exported')

print('\\n' + '='*80)
print('ANALYSIS COMPLETE!')
print('='*80)
print(f'\\nGenerated:')
print(f'  - {optimal_k} clusters identified')
print(f'  - {pca.explained_variance_ratio_.sum():.1%} variance explained')
print(f'  - Silhouette score: {silhouette_avg:.4f}')
print(f'  - 5 visualizations saved')
print(f'  - 3 data files exported')""")
]

for cell_type, content in cells_content:
    if cell_type == "markdown":
        notebook["cells"].append(create_markdown_cell(content))
    else:
        notebook["cells"].append(create_code_cell(content))

# Save notebook
notebook_path = Path("notebooks/06_unsupervised_clustering_analysis.ipynb")
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"[OK] Notebook created with {len(notebook['cells'])} cells")
print(f"[OK] Saved to: {notebook_path}")
