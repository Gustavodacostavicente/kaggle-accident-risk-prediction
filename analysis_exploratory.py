"""
Exploratory Data Analysis for Kaggle Accident Risk Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
print("Loading train.csv...")
df = pd.read_csv("train.csv")

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Create output directory for plots
output_dir = Path("eda_plots")
output_dir.mkdir(exist_ok=True)

# ============================================================================
# 1. BASIC STATISTICS
# ============================================================================
print("\n" + "="*80)
print("BASIC STATISTICS")
print("="*80)

print("\nDataset Info:")
print(df.info())

print("\nDescriptive Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

# ============================================================================
# 2. TARGET VARIABLE ANALYSIS (accident_risk)
# ============================================================================
print("\n" + "="*80)
print("TARGET VARIABLE ANALYSIS")
print("="*80)

target = "accident_risk"
if target in df.columns:
    print(f"\n{target} statistics:")
    print(df[target].describe())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Histogram
    axes[0].hist(df[target], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel(target)
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Distribution of {target}')
    axes[0].axvline(df[target].mean(), color='red', linestyle='--', label=f'Mean: {df[target].mean():.2f}')
    axes[0].axvline(df[target].median(), color='green', linestyle='--', label=f'Median: {df[target].median():.2f}')
    axes[0].legend()

    # Box plot
    axes[1].boxplot(df[target], vert=True)
    axes[1].set_ylabel(target)
    axes[1].set_title(f'Box Plot of {target}')
    axes[1].grid(True, alpha=0.3)

    # KDE plot
    df[target].plot(kind='kde', ax=axes[2], linewidth=2)
    axes[2].set_xlabel(target)
    axes[2].set_title(f'Density Plot of {target}')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "01_target_distribution.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/01_target_distribution.png")
    plt.close()

# ============================================================================
# 3. NUMERIC FEATURES DISTRIBUTION
# ============================================================================
print("\n" + "="*80)
print("NUMERIC FEATURES DISTRIBUTION")
print("="*80)

# Identify numeric columns (excluding id and target)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['id', target]]

print(f"\nNumeric features: {numeric_cols}")

if numeric_cols:
    # Histograms for all numeric features
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for idx, col in enumerate(numeric_cols):
        axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "02_numeric_histograms.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/02_numeric_histograms.png")
    plt.close()

    # Box plots for all numeric features
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for idx, col in enumerate(numeric_cols):
        axes[idx].boxplot(df[col].dropna(), vert=True)
        axes[idx].set_ylabel(col)
        axes[idx].set_title(f'Box Plot of {col}')
        axes[idx].grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "03_numeric_boxplots.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/03_numeric_boxplots.png")
    plt.close()

# ============================================================================
# 4. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

if numeric_cols and target in df.columns:
    # Correlation matrix
    corr_cols = numeric_cols + [target]
    corr_matrix = df[corr_cols].corr()

    print("\nCorrelation with target:")
    target_corr = corr_matrix[target].sort_values(ascending=False)
    print(target_corr)

    # Heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1)
    plt.title('Correlation Matrix Heatmap', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "04_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/04_correlation_heatmap.png")
    plt.close()

    # Correlation with target bar plot
    target_corr_abs = target_corr[target_corr.index != target].abs().sort_values(ascending=True)

    plt.figure(figsize=(10, max(6, len(target_corr_abs) * 0.3)))
    colors = ['red' if x < 0 else 'green' for x in target_corr[target_corr_abs.index]]
    target_corr_abs.plot(kind='barh', color=colors, alpha=0.7)
    plt.xlabel('Absolute Correlation with accident_risk')
    plt.title('Feature Correlation with Target (Absolute Values)')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_dir / "05_target_correlation_bars.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/05_target_correlation_bars.png")
    plt.close()

# ============================================================================
# 5. SCATTER PLOTS - TOP CORRELATED FEATURES vs TARGET
# ============================================================================
print("\n" + "="*80)
print("SCATTER PLOTS - TOP FEATURES vs TARGET")
print("="*80)

if numeric_cols and target in df.columns:
    # Get top 9 most correlated features
    top_features = target_corr[target_corr.index != target].abs().sort_values(ascending=False).head(9).index.tolist()

    if top_features:
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        axes = axes.flatten()

        for idx, feature in enumerate(top_features):
            axes[idx].scatter(df[feature], df[target], alpha=0.5, s=10)
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel(target)
            corr_val = corr_matrix.loc[feature, target]
            axes[idx].set_title(f'{feature} vs {target}\nCorrelation: {corr_val:.3f}')
            axes[idx].grid(True, alpha=0.3)

            # Add trend line
            z = np.polyfit(df[feature].dropna(), df[target][df[feature].notna()], 1)
            p = np.poly1d(z)
            axes[idx].plot(df[feature], p(df[feature]), "r--", alpha=0.8, linewidth=2)

        # Hide unused subplots
        for idx in range(len(top_features), 9):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_dir / "06_scatter_top_features.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/06_scatter_top_features.png")
        plt.close()

# ============================================================================
# 6. PAIRPLOT - TOP 5 FEATURES
# ============================================================================
print("\n" + "="*80)
print("PAIRPLOT - TOP 5 FEATURES")
print("="*80)

if numeric_cols and target in df.columns and len(numeric_cols) >= 5:
    top_5_features = target_corr[target_corr.index != target].abs().sort_values(ascending=False).head(5).index.tolist()

    if top_5_features:
        print(f"Creating pairplot for: {top_5_features + [target]}")

        pairplot_df = df[top_5_features + [target]].copy()
        sns.pairplot(pairplot_df, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 10})
        plt.suptitle('Pairplot - Top 5 Features vs Target', y=1.01, fontsize=16)
        plt.savefig(output_dir / "07_pairplot_top5.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/07_pairplot_top5.png")
        plt.close()

# ============================================================================
# 7. CATEGORICAL FEATURES ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("CATEGORICAL FEATURES ANALYSIS")
print("="*80)

categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_cols = [col for col in categorical_cols if col != 'id']

print(f"\nCategorical features: {categorical_cols}")

if categorical_cols:
    for col in categorical_cols:
        print(f"\n{col} - Value counts:")
        print(df[col].value_counts())

        # Bar plot
        plt.figure(figsize=(12, 6))
        df[col].value_counts().plot(kind='bar', color='steelblue', edgecolor='black')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / f"08_categorical_{col}.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/08_categorical_{col}.png")
        plt.close()

        # Box plot: target by category
        if target in df.columns:
            plt.figure(figsize=(12, 6))
            df.boxplot(column=target, by=col, ax=plt.gca())
            plt.xlabel(col)
            plt.ylabel(target)
            plt.title(f'{target} by {col}')
            plt.suptitle('')  # Remove automatic title
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / f"09_target_by_{col}.png", dpi=300, bbox_inches='tight')
            print(f"Saved: {output_dir}/09_target_by_{col}.png")
            plt.close()

# ============================================================================
# 8. OUTLIER ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("OUTLIER ANALYSIS")
print("="*80)

if numeric_cols:
    print("\nOutliers detected (using IQR method):")

    outlier_summary = {}

    for col in numeric_cols + ([target] if target in df.columns else []):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_summary[col] = {
            'count': len(outliers),
            'percentage': len(outliers) / len(df) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

        print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")

    # Visualize outlier counts
    outlier_counts = pd.Series({k: v['count'] for k, v in outlier_summary.items()})
    outlier_counts = outlier_counts.sort_values(ascending=True)

    plt.figure(figsize=(10, max(6, len(outlier_counts) * 0.3)))
    outlier_counts.plot(kind='barh', color='coral', edgecolor='black')
    plt.xlabel('Number of Outliers')
    plt.title('Outlier Count by Feature (IQR Method)')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_dir / "10_outlier_counts.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/10_outlier_counts.png")
    plt.close()

# ============================================================================
# 9. STATISTICAL SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("GENERATING STATISTICAL SUMMARY REPORT")
print("="*80)

with open(output_dir / "statistical_summary.txt", "w") as f:
    f.write("="*80 + "\n")
    f.write("EXPLORATORY DATA ANALYSIS SUMMARY\n")
    f.write("="*80 + "\n\n")

    f.write(f"Dataset shape: {df.shape}\n")
    f.write(f"Number of features: {len(df.columns) - 1}\n")
    f.write(f"Number of samples: {len(df)}\n\n")

    f.write("Missing Values:\n")
    f.write(str(df.isnull().sum()) + "\n\n")

    f.write("Descriptive Statistics:\n")
    f.write(str(df.describe()) + "\n\n")

    if target in df.columns:
        f.write(f"\nTarget Variable ({target}) Statistics:\n")
        f.write(str(df[target].describe()) + "\n\n")

    if numeric_cols and target in df.columns:
        f.write("\nTop 10 Features Correlated with Target:\n")
        f.write(str(target_corr.head(11)) + "\n\n")

    f.write("\nOutlier Summary:\n")
    for col, info in outlier_summary.items():
        f.write(f"{col}: {info['count']} outliers ({info['percentage']:.2f}%)\n")

print(f"Saved: {output_dir}/statistical_summary.txt")

print("\n" + "="*80)
print("EDA COMPLETE!")
print("="*80)
print(f"\nAll plots saved to: {output_dir}/")
print("\nGenerated files:")
for file in sorted(output_dir.glob("*")):
    print(f"  - {file.name}")
