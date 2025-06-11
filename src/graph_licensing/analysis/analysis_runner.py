"""Modern analysis runner with improved visualizations."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Modern color palette with blue and red as primary colors
COLORS = {
    'primary_blue': '#2E86AB',
    'primary_red': '#A23B72', 
    'secondary_blue': '#A7C5EB',
    'secondary_red': '#F18F89',
    'accent': '#F5F3F0',
    'dark': '#2C2C2C',
    'light_gray': '#E8E8E8'
}

# Set modern matplotlib style
plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.linewidth': 0.8,
    'axes.labelcolor': COLORS['dark'],
    'text.color': COLORS['dark'],
    'xtick.color': COLORS['dark'],
    'ytick.color': COLORS['dark'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


class AnalysisRunner:
    """Modern analysis runner with consistent blue-red color scheme."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.color_palette = [COLORS['primary_blue'], COLORS['primary_red'], 
                             COLORS['secondary_blue'], COLORS['secondary_red']]
        
    def run_full_analysis(
        self, 
        input_path: str, 
        output_path: str,
        file_pattern: str = "*.json"
    ) -> None:
        """Run comprehensive analysis."""
        
        input_dir = Path(input_path)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        data = self._load_benchmark_data(input_dir, file_pattern)
        if not data:
            self.logger.error("No data found to analyze")
            return
            
        df = pd.DataFrame(data)
        
        # Generate visualizations
        self._create_algorithm_comparison(df, output_dir)
        self._create_performance_analysis(df, output_dir)
        self._create_scalability_analysis(df, output_dir)
        self._create_quality_analysis(df, output_dir)
        self._create_efficiency_analysis(df, output_dir)
        self._create_correlation_analysis(df, output_dir)
        
        # Generate summary report
        self._generate_summary_report(df, output_dir)
        
        self.logger.info(f"Analysis complete. Results saved to {output_dir}")
    
    def _load_benchmark_data(self, input_dir: Path, pattern: str) -> List[Dict]:
        """Load benchmark data from JSON files."""
        data = []
        
        for file_path in input_dir.glob(pattern):
            try:
                with open(file_path, 'r') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        data.extend(file_data)
                    else:
                        data.append(file_data)
            except Exception as e:
                self.logger.warning(f"Failed to load {file_path}: {e}")
        
        return data
    
    def _create_algorithm_comparison(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create algorithm comparison visualizations."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
        
        # Cost comparison
        if 'total_cost' in df.columns and 'algorithm' in df.columns:
            df_cost = df.groupby('algorithm')['total_cost'].agg(['mean', 'std']).reset_index()
            bars = axes[0,0].bar(df_cost['algorithm'], df_cost['mean'], 
                               color=self.color_palette, alpha=0.8)
            axes[0,0].errorbar(df_cost['algorithm'], df_cost['mean'], 
                             yerr=df_cost['std'], fmt='none', color='black', capsize=3)
            axes[0,0].set_title('Average Total Cost by Algorithm', fontweight='bold')
            axes[0,0].set_ylabel('Total Cost')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, df_cost['mean']):
                axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + df_cost['std'].max()*0.1,
                             f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Runtime comparison
        if 'runtime_seconds' in df.columns:
            df_runtime = df.groupby('algorithm')['runtime_seconds'].agg(['mean', 'std']).reset_index()
            bars = axes[0,1].bar(df_runtime['algorithm'], df_runtime['mean'], 
                               color=self.color_palette, alpha=0.8)
            axes[0,1].errorbar(df_runtime['algorithm'], df_runtime['mean'], 
                             yerr=df_runtime['std'], fmt='none', color='black', capsize=3)
            axes[0,1].set_title('Average Runtime by Algorithm', fontweight='bold')
            axes[0,1].set_ylabel('Runtime (seconds)')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, df_runtime['mean']):
                axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + df_runtime['std'].max()*0.1,
                             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Success rate
        if 'success' in df.columns:
            success_rate = df.groupby('algorithm')['success'].mean().reset_index()
            bars = axes[1,0].bar(success_rate['algorithm'], success_rate['success'] * 100, 
                               color=self.color_palette, alpha=0.8)
            axes[1,0].set_title('Success Rate by Algorithm', fontweight='bold')
            axes[1,0].set_ylabel('Success Rate (%)')
            axes[1,0].set_ylim(0, 105)
            axes[1,0].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, success_rate['success']):
                axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                             f'{value*100:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Group formation rate
        if 'n_group_licenses' in df.columns and 'n_solo_licenses' in df.columns:
            df['group_ratio'] = df['n_group_licenses'] / (df['n_group_licenses'] + df['n_solo_licenses'])
            group_ratio = df.groupby('algorithm')['group_ratio'].mean().reset_index()
            bars = axes[1,1].bar(group_ratio['algorithm'], group_ratio['group_ratio'] * 100, 
                               color=self.color_palette, alpha=0.8)
            axes[1,1].set_title('Group License Formation Rate', fontweight='bold')
            axes[1,1].set_ylabel('Group Formation Rate (%)')
            axes[1,1].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, group_ratio['group_ratio']):
                axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                             f'{value*100:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'algorithm_comparison.png')
        plt.close()
    
    def _create_performance_analysis(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create performance analysis visualizations."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Analysis', fontsize=16, fontweight='bold')
        
        # Cost vs Runtime scatter
        if 'total_cost' in df.columns and 'runtime_seconds' in df.columns:
            algorithms = df['algorithm'].unique()
            for i, algo in enumerate(algorithms):
                algo_data = df[df['algorithm'] == algo]
                color = self.color_palette[i % len(self.color_palette)]
                axes[0,0].scatter(algo_data['runtime_seconds'], algo_data['total_cost'], 
                                label=algo, alpha=0.7, color=color, s=50)
            
            axes[0,0].set_xlabel('Runtime (seconds)')
            axes[0,0].set_ylabel('Total Cost')
            axes[0,0].set_title('Cost vs Runtime Trade-off', fontweight='bold')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
        
        # Cost distribution
        if 'total_cost' in df.columns and 'algorithm' in df.columns:
            algorithms = df['algorithm'].unique()
            for i, algo in enumerate(algorithms):
                algo_data = df[df['algorithm'] == algo]['total_cost']
                color = self.color_palette[i % len(self.color_palette)]
                axes[0,1].hist(algo_data, alpha=0.6, label=algo, color=color, bins=20)
            
            axes[0,1].set_xlabel('Total Cost')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].set_title('Cost Distribution by Algorithm', fontweight='bold')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # Runtime distribution
        if 'runtime_seconds' in df.columns:
            algorithms = df['algorithm'].unique()
            for i, algo in enumerate(algorithms):
                algo_data = df[df['algorithm'] == algo]['runtime_seconds']
                color = self.color_palette[i % len(self.color_palette)]
                axes[1,0].hist(algo_data, alpha=0.6, label=algo, color=color, bins=20)
            
            axes[1,0].set_xlabel('Runtime (seconds)')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_title('Runtime Distribution by Algorithm', fontweight='bold')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # Efficiency plot (cost per time)
        if 'total_cost' in df.columns and 'runtime_seconds' in df.columns:
            df['efficiency'] = df['total_cost'] / (df['runtime_seconds'] + 1e-6)
            efficiency = df.groupby('algorithm')['efficiency'].mean().reset_index()
            bars = axes[1,1].bar(efficiency['algorithm'], efficiency['efficiency'], 
                               color=self.color_palette, alpha=0.8)
            axes[1,1].set_title('Cost Efficiency (Cost/Time)', fontweight='bold')
            axes[1,1].set_ylabel('Cost per Second')
            axes[1,1].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, efficiency['efficiency']):
                axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + efficiency['efficiency'].max()*0.02,
                             f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_analysis.png')
        plt.close()
    
    def _create_scalability_analysis(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create scalability analysis."""
        
        if 'n_nodes' not in df.columns:
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Scalability Analysis', fontsize=16, fontweight='bold')
        
        algorithms = df['algorithm'].unique()
        
        # Runtime vs graph size
        if 'runtime_seconds' in df.columns:
            for i, algo in enumerate(algorithms):
                algo_data = df[df['algorithm'] == algo]
                if not algo_data.empty:
                    grouped = algo_data.groupby('n_nodes')['runtime_seconds'].mean().reset_index()
                    color = self.color_palette[i % len(self.color_palette)]
                    axes[0].plot(grouped['n_nodes'], grouped['runtime_seconds'], 
                               marker='o', label=algo, color=color, linewidth=2, markersize=6)
            
            axes[0].set_xlabel('Number of Nodes')
            axes[0].set_ylabel('Runtime (seconds)')
            axes[0].set_title('Runtime Scalability', fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_yscale('log')
        
        # Cost vs graph size
        if 'total_cost' in df.columns:
            for i, algo in enumerate(algorithms):
                algo_data = df[df['algorithm'] == algo]
                if not algo_data.empty:
                    grouped = algo_data.groupby('n_nodes')['total_cost'].mean().reset_index()
                    color = self.color_palette[i % len(self.color_palette)]
                    axes[1].plot(grouped['n_nodes'], grouped['total_cost'], 
                               marker='o', label=algo, color=color, linewidth=2, markersize=6)
            
            axes[1].set_xlabel('Number of Nodes')
            axes[1].set_ylabel('Total Cost')
            axes[1].set_title('Cost Scalability', fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'scalability_analysis.png')
        plt.close()
    
    def _create_quality_analysis(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create solution quality analysis."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Solution Quality Analysis', fontsize=16, fontweight='bold')
        
        # Average group size
        if 'avg_group_size' in df.columns:
            avg_group_size = df.groupby('algorithm')['avg_group_size'].mean().reset_index()
            bars = axes[0,0].bar(avg_group_size['algorithm'], avg_group_size['avg_group_size'], 
                               color=self.color_palette, alpha=0.8)
            axes[0,0].set_title('Average Group Size', fontweight='bold')
            axes[0,0].set_ylabel('Average Group Size')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, avg_group_size['avg_group_size']):
                axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                             f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Valid solutions percentage
        if 'is_valid_solution' in df.columns:
            validity = df.groupby('algorithm')['is_valid_solution'].mean().reset_index()
            bars = axes[0,1].bar(validity['algorithm'], validity['is_valid_solution'] * 100, 
                               color=self.color_palette, alpha=0.8)
            axes[0,1].set_title('Solution Validity Rate', fontweight='bold')
            axes[0,1].set_ylabel('Valid Solutions (%)')
            axes[0,1].set_ylim(0, 105)
            axes[0,1].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, validity['is_valid_solution']):
                axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                             f'{value*100:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Group vs Solo ratio
        if 'n_group_licenses' in df.columns and 'n_solo_licenses' in df.columns:
            df_copy = df.copy()
            df_copy['total_licenses'] = df_copy['n_group_licenses'] + df_copy['n_solo_licenses']
            df_copy['group_percentage'] = (df_copy['n_group_licenses'] / df_copy['total_licenses']) * 100
            
            group_pct = df_copy.groupby('algorithm')['group_percentage'].mean().reset_index()
            bars = axes[1,0].bar(group_pct['algorithm'], group_pct['group_percentage'], 
                               color=self.color_palette, alpha=0.8)
            axes[1,0].set_title('Group License Percentage', fontweight='bold')
            axes[1,0].set_ylabel('Group Licenses (%)')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, group_pct['group_percentage']):
                axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                             f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Cost efficiency (lower is better)
        if 'total_cost' in df.columns and 'n_nodes' in df.columns:
            df_copy = df.copy()
            df_copy['cost_per_node'] = df_copy['total_cost'] / df_copy['n_nodes']
            cost_eff = df_copy.groupby('algorithm')['cost_per_node'].mean().reset_index()
            bars = axes[1,1].bar(cost_eff['algorithm'], cost_eff['cost_per_node'], 
                               color=self.color_palette, alpha=0.8)
            axes[1,1].set_title('Cost per Node', fontweight='bold')
            axes[1,1].set_ylabel('Cost per Node')
            axes[1,1].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, cost_eff['cost_per_node']):
                axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + cost_eff['cost_per_node'].max()*0.02,
                             f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'quality_analysis.png')
        plt.close()
    
    def _create_efficiency_analysis(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create efficiency analysis."""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Algorithm Efficiency Analysis', fontsize=16, fontweight='bold')
        
        # Time efficiency (solutions per second)
        if 'runtime_seconds' in df.columns:
            df_copy = df.copy()
            df_copy['solutions_per_second'] = 1 / (df_copy['runtime_seconds'] + 1e-6)
            time_eff = df_copy.groupby('algorithm')['solutions_per_second'].mean().reset_index()
            bars = axes[0].bar(time_eff['algorithm'], time_eff['solutions_per_second'], 
                             color=self.color_palette, alpha=0.8)
            axes[0].set_title('Time Efficiency (Solutions/Second)', fontweight='bold')
            axes[0].set_ylabel('Solutions per Second')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].set_yscale('log')
            
            for bar, value in zip(bars, time_eff['solutions_per_second']):
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                             f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Overall efficiency score (normalized)
        if 'total_cost' in df.columns and 'runtime_seconds' in df.columns:
            # Normalize metrics (lower is better for both)
            df_norm = df.copy()
            algorithms = df_norm['algorithm'].unique()
            
            efficiency_scores = []
            for algo in algorithms:
                algo_data = df_norm[df_norm['algorithm'] == algo]
                
                # Normalize cost (0-1, lower is better)
                cost_norm = (algo_data['total_cost'] - df_norm['total_cost'].min()) / \
                           (df_norm['total_cost'].max() - df_norm['total_cost'].min() + 1e-6)
                
                # Normalize runtime (0-1, lower is better)
                runtime_norm = (algo_data['runtime_seconds'] - df_norm['runtime_seconds'].min()) / \
                              (df_norm['runtime_seconds'].max() - df_norm['runtime_seconds'].min() + 1e-6)
                
                # Combined efficiency (lower is better)
                efficiency = (cost_norm + runtime_norm) / 2
                efficiency_scores.append({
                    'algorithm': algo,
                    'efficiency_score': 1 - efficiency.mean()  # Invert so higher is better
                })
            
            eff_df = pd.DataFrame(efficiency_scores)
            bars = axes[1].bar(eff_df['algorithm'], eff_df['efficiency_score'], 
                             color=self.color_palette, alpha=0.8)
            axes[1].set_title('Overall Efficiency Score', fontweight='bold')
            axes[1].set_ylabel('Efficiency Score (0-1)')
            axes[1].set_ylim(0, 1.1)
            axes[1].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, eff_df['efficiency_score']):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'efficiency_analysis.png')
        plt.close()
    
    def _create_correlation_analysis(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Create correlation analysis."""
        
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_cols = [col for col in numeric_cols if col in [
            'total_cost', 'runtime_seconds', 'n_nodes', 'n_edges', 
            'n_solo_licenses', 'n_group_licenses', 'avg_group_size'
        ]]
        
        if len(correlation_cols) < 2:
            return
        
        plt.figure(figsize=(12, 10))
        
        # Calculate correlation matrix
        corr_matrix = df[correlation_cols].corr()
        
        # Create heatmap with blue-red colormap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(220, 20, as_cmap=True)  # Blue to red
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                   square=True, annot=True, fmt='.2f', cbar_kws={"shrink": .8})
        
        plt.title('Correlation Matrix of Key Metrics', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_analysis.png')
        plt.close()
    
    def _generate_summary_report(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Generate summary report."""
        
        report = []
        report.append("=== GRAPH LICENSING OPTIMIZATION ANALYSIS REPORT ===\\n")
        
        # Dataset overview
        report.append("DATASET OVERVIEW:")
        report.append(f"Total experiments: {len(df)}")
        report.append(f"Algorithms tested: {', '.join(df['algorithm'].unique())}")
        report.append(f"Graph sizes: {df['n_nodes'].min()}-{df['n_nodes'].max()} nodes")
        report.append("")
        
        # Algorithm performance summary
        report.append("ALGORITHM PERFORMANCE SUMMARY:")
        if 'total_cost' in df.columns:
            cost_summary = df.groupby('algorithm')['total_cost'].agg(['mean', 'std', 'min', 'max'])
            for algo in cost_summary.index:
                stats = cost_summary.loc[algo]
                report.append(f"{algo}:")
                report.append(f"  Average cost: {stats['mean']:.2f} ± {stats['std']:.2f}")
                report.append(f"  Cost range: {stats['min']:.2f} - {stats['max']:.2f}")
        
        report.append("")
        
        # Runtime performance
        if 'runtime_seconds' in df.columns:
            report.append("RUNTIME PERFORMANCE:")
            runtime_summary = df.groupby('algorithm')['runtime_seconds'].agg(['mean', 'std'])
            for algo in runtime_summary.index:
                stats = runtime_summary.loc[algo]
                report.append(f"{algo}: {stats['mean']:.4f}s ± {stats['std']:.4f}s")
        
        report.append("")
        
        # Success rates
        if 'success' in df.columns:
            report.append("SUCCESS RATES:")
            success_rates = df.groupby('algorithm')['success'].mean() * 100
            for algo, rate in success_rates.items():
                report.append(f"{algo}: {rate:.1f}%")
        
        report.append("")
        
        # Group formation analysis
        if 'n_group_licenses' in df.columns and 'n_solo_licenses' in df.columns:
            report.append("GROUP FORMATION ANALYSIS:")
            df_temp = df.copy()
            df_temp['group_ratio'] = df_temp['n_group_licenses'] / (
                df_temp['n_group_licenses'] + df_temp['n_solo_licenses']
            )
            group_stats = df_temp.groupby('algorithm')['group_ratio'].mean() * 100
            for algo, ratio in group_stats.items():
                report.append(f"{algo}: {ratio:.1f}% group licenses")
        
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        
        if 'total_cost' in df.columns:
            best_cost_algo = df.groupby('algorithm')['total_cost'].mean().idxmin()
            report.append(f"• Best cost performance: {best_cost_algo}")
        
        if 'runtime_seconds' in df.columns:
            fastest_algo = df.groupby('algorithm')['runtime_seconds'].mean().idxmin()
            report.append(f"• Fastest algorithm: {fastest_algo}")
        
        if 'success' in df.columns:
            most_reliable = df.groupby('algorithm')['success'].mean().idxmax()
            report.append(f"• Most reliable: {most_reliable}")
        
        # Save report
        with open(output_dir / 'analysis_summary.txt', 'w') as f:
            f.write('\\n'.join(report))
        
        self.logger.info("Summary report generated")
