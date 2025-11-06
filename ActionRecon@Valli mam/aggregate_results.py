#!/usr/bin/env python3
"""
Results Aggregation Script for HAR Models
This script aggregates results from all four models and generates comprehensive comparison tables and visualizations.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class ResultsAggregator:
    def __init__(self):
        self.models = ['lstm', 'resnet_transformer', 'tcn', 'transformer']
        self.results = {}
        self.load_all_results()
    
    def load_all_results(self):
        """Load results from all model JSON files"""
        print("Loading results from all models...")
        
        for model in self.models:
            try:
                with open(f'{model}_results.json', 'r') as f:
                    self.results[model] = json.load(f)
                print(f"✓ Loaded {model} results")
            except FileNotFoundError:
                print(f"✗ {model}_results.json not found")
                self.results[model] = None
    
    def create_performance_summary_table(self):
        """Create comprehensive performance summary table"""
        print("\nCreating performance summary table...")
        
        summary_data = []
        for model in self.models:
            if self.results[model] is not None:
                summary_data.append({
                    'Model': model.replace('_', '-').title(),
                    'Accuracy (%)': round(self.results[model]['accuracy'] * 100, 2),
                    'Precision (%)': round(self.results[model]['precision'] * 100, 2),
                    'Recall (%)': round(self.results[model]['recall'] * 100, 2),
                    'F1-Score (%)': round(self.results[model]['f1_score'] * 100, 2),
                    'AUC (%)': round(self.results[model]['auc'] * 100, 2)
                })
        
        summary_df = pd.DataFrame(summary_data)
        print("\n=== PERFORMANCE SUMMARY TABLE ===")
        print(summary_df.to_string(index=False))
        
        # Save to CSV
        summary_df.to_csv('performance_summary.csv', index=False)
        print("\nPerformance summary saved to 'performance_summary.csv'")
        
        return summary_df
    
    def create_per_class_comparison(self):
        """Create per-class performance comparison"""
        print("\nCreating per-class performance comparison...")
        
        all_class_reports = {}
        for model in self.models:
            if self.results[model] is not None:
                class_report = self.results[model]['classification_report']
                all_class_reports[model] = class_report
        
        # Get all unique classes
        all_classes = set()
        for model_reports in all_class_reports.values():
            all_classes.update([k for k in model_reports.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']])
        
        # Create comparison table
        comparison_data = []
        for class_name in sorted(all_classes):
            row = {'Class': class_name}
            for model in self.models:
                if model in all_class_reports and class_name in all_class_reports[model]:
                    metrics = all_class_reports[model][class_name]
                    row[f'{model}_precision'] = round(metrics['precision'] * 100, 2)
                    row[f'{model}_recall'] = round(metrics['recall'] * 100, 2)
                    row[f'{model}_f1'] = round(metrics['f1-score'] * 100, 2)
                else:
                    row[f'{model}_precision'] = 0
                    row[f'{model}_recall'] = 0
                    row[f'{model}_f1'] = 0
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n=== PER-CLASS PERFORMANCE COMPARISON ===")
        print(comparison_df.to_string(index=False))
        
        # Save to CSV
        comparison_df.to_csv('per_class_comparison.csv', index=False)
        print("\nPer-class comparison saved to 'per_class_comparison.csv'")
        
        return comparison_df
    
    def create_confusion_matrices_comparison(self):
        """Create side-by-side confusion matrices comparison"""
        print("\nCreating confusion matrices comparison...")
        
        # Load class names from one of the models
        class_names = None
        for model in self.models:
            if self.results[model] is not None:
                # Try to get class names from classification report
                class_report = self.results[model]['classification_report']
                class_names = [k for k in class_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
                break
        
        if class_names is None:
            print("No class names found")
            return
        
        # Create subplot for confusion matrices
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, model in enumerate(self.models):
            if self.results[model] is not None:
                cm = np.array(self.results[model]['confusion_matrix'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=class_names, yticklabels=class_names,
                           ax=axes[i])
                axes[i].set_title(f'{model.replace("_", "-").title()} Confusion Matrix', fontweight='bold')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
            else:
                axes[i].text(0.5, 0.5, f'{model} results not available', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{model.replace("_", "-").title()} - No Data')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Confusion matrices comparison saved to 'confusion_matrices_comparison.png'")
    
    def create_performance_radar_chart(self):
        """Create radar chart comparing model performance"""
        print("\nCreating performance radar chart...")
        
        # Prepare data for radar chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        model_data = {}
        
        for model in self.models:
            if self.results[model] is not None:
                model_data[model.replace('_', '-').title()] = [
                    self.results[model]['accuracy'] * 100,
                    self.results[model]['precision'] * 100,
                    self.results[model]['recall'] * 100,
                    self.results[model]['f1_score'] * 100,
                    self.results[model]['auc'] * 100
                ]
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Calculate angles for each metric
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, (model_name, values) in enumerate(model_data.items()):
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.grid(True)
        
        plt.title('Model Performance Comparison - Radar Chart', size=16, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig('performance_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Performance radar chart saved to 'performance_radar_chart.png'")
    
    def create_bar_chart_comparison(self):
        """Create bar chart comparing key metrics"""
        print("\nCreating bar chart comparison...")
        
        # Prepare data
        models = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        aucs = []
        
        for model in self.models:
            if self.results[model] is not None:
                models.append(model.replace('_', '-').title())
                accuracies.append(self.results[model]['accuracy'] * 100)
                precisions.append(self.results[model]['precision'] * 100)
                recalls.append(self.results[model]['recall'] * 100)
                f1_scores.append(self.results[model]['f1_score'] * 100)
                aucs.append(self.results[model]['auc'] * 100)
        
        # Create bar chart
        x = np.arange(len(models))
        width = 0.15
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.bar(x - 2*width, accuracies, width, label='Accuracy', alpha=0.8)
        ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        ax.bar(x + 2*width, aucs, width, label='AUC', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Performance (%)')
        ax.set_title('Model Performance Comparison - Bar Chart', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, model in enumerate(models):
            ax.text(i - 2*width, accuracies[i] + 1, f'{accuracies[i]:.1f}', ha='center', va='bottom', fontsize=8)
            ax.text(i - width, precisions[i] + 1, f'{precisions[i]:.1f}', ha='center', va='bottom', fontsize=8)
            ax.text(i, recalls[i] + 1, f'{recalls[i]:.1f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + width, f1_scores[i] + 1, f'{f1_scores[i]:.1f}', ha='center', va='bottom', fontsize=8)
            ax.text(i + 2*width, aucs[i] + 1, f'{aucs[i]:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('performance_bar_chart.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Performance bar chart saved to 'performance_bar_chart.png'")
    
    def generate_statistical_analysis(self):
        """Generate statistical analysis of results"""
        print("\nGenerating statistical analysis...")
        
        # Collect all metrics for statistical analysis
        metrics_data = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
            values = []
            for model in self.models:
                if self.results[model] is not None:
                    values.append(self.results[model][metric] * 100)
            metrics_data[metric] = values
        
        # Create statistical summary
        stats_summary = {}
        for metric, values in metrics_data.items():
            if values:
                stats_summary[metric] = {
                    'Mean': round(np.mean(values), 2),
                    'Std': round(np.std(values), 2),
                    'Min': round(np.min(values), 2),
                    'Max': round(np.max(values), 2),
                    'Range': round(np.max(values) - np.min(values), 2)
                }
        
        stats_df = pd.DataFrame(stats_summary).T
        print("\n=== STATISTICAL ANALYSIS ===")
        print(stats_df.to_string())
        
        # Save statistical analysis
        stats_df.to_csv('statistical_analysis.csv')
        print("\nStatistical analysis saved to 'statistical_analysis.csv'")
        
        return stats_df
    
    def create_final_report(self):
        """Create final comprehensive report"""
        print("\nCreating final comprehensive report...")
        
        report = []
        report.append("# HAR Models - Comprehensive Results Report")
        report.append("=" * 50)
        report.append("")
        
        # Performance Summary
        report.append("## Performance Summary")
        report.append("")
        summary_df = self.create_performance_summary_table()
        report.append(summary_df.to_string(index=False))
        report.append("")
        
        # Best performing model
        best_model = summary_df.loc[summary_df['Accuracy (%)'].idxmax(), 'Model']
        best_accuracy = summary_df['Accuracy (%)'].max()
        report.append(f"**Best Performing Model**: {best_model} with {best_accuracy}% accuracy")
        report.append("")
        
        # Statistical Analysis
        report.append("## Statistical Analysis")
        report.append("")
        stats_df = self.generate_statistical_analysis()
        report.append(stats_df.to_string())
        report.append("")
        
        # Key Findings
        report.append("## Key Findings")
        report.append("")
        report.append("1. **Model Performance Ranking**:")
        sorted_models = summary_df.sort_values('Accuracy (%)', ascending=False)
        for i, (_, row) in enumerate(sorted_models.iterrows(), 1):
            report.append(f"   {i}. {row['Model']}: {row['Accuracy (%)']}% accuracy")
        report.append("")
        
        report.append("2. **Performance Consistency**:")
        for metric in ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'AUC (%)']:
            std_dev = summary_df[metric].std()
            report.append(f"   - {metric}: Standard deviation = {std_dev:.2f}%")
        report.append("")
        
        # Save report
        with open('comprehensive_results_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        print("Comprehensive report saved to 'comprehensive_results_report.md'")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("=== HAR Models - Results Aggregation ===")
        print("Starting comprehensive analysis...")
        
        # Check if all results are available
        available_models = [model for model in self.models if self.results[model] is not None]
        print(f"Available models: {available_models}")
        
        if not available_models:
            print("No model results found. Please run the individual model scripts first.")
            return
        
        # Run all analysis components
        self.create_performance_summary_table()
        self.create_per_class_comparison()
        self.create_confusion_matrices_comparison()
        self.create_performance_radar_chart()
        self.create_bar_chart_comparison()
        self.generate_statistical_analysis()
        self.create_final_report()
        
        print("\n=== Analysis Complete ===")
        print("Generated files:")
        print("- performance_summary.csv")
        print("- per_class_comparison.csv")
        print("- confusion_matrices_comparison.png")
        print("- performance_radar_chart.png")
        print("- performance_bar_chart.png")
        print("- statistical_analysis.csv")
        print("- comprehensive_results_report.md")

def main():
    """Main execution function"""
    aggregator = ResultsAggregator()
    aggregator.run_full_analysis()

if __name__ == "__main__":
    main()
