# analyze_results.py
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List
import seaborn as sns

def analyze_experiment(exp_dir: str):
    """تحلیل نتایج آزمایش"""
    
    # بارگذاری نتایج
    stats_path = os.path.join(exp_dir, "results", "statistics.json")
    comparison_path = os.path.join(exp_dir, "results", "comparison.json")
    
    if not os.path.exists(stats_path):
        print(f"Statistics file not found: {stats_path}")
        return
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    if os.path.exists(comparison_path):
        with open(comparison_path, 'r') as f:
            comparison = json.load(f)
    else:
        comparison = {}
    
    # نمایش نتایج
    print("\n" + "="*60)
    print("EXPERIMENT ANALYSIS")
    print("="*60)
    
    print(f"\nOptimization Results:")
    print(f"  Seed SP: {stats['seed_sp']:.2f}")
    print(f"  Optimized SP: {stats['optimized_sp']:.2f}")
    print(f"  Refined SP: {stats['refined_sp']:.2f}")
    print(f"  Total Improvement: {stats['improvement']:.2f}")
    print(f"  Improvement Percentage: {stats['improvement']/stats['seed_sp']*100:.1f}%")
    
    print(f"\nOptimal Parameters:")
    for param, value in stats['optimal_params'].items():
        print(f"  {param}: {value}")
    
    if comparison:
        print(f"\nMethod Comparison:")
        methods = list(comparison.keys())
        sp_scores = [comparison[m]['sp'] for m in methods]
        
        for method, sp in zip(methods, sp_scores):
            print(f"  {method:<20}: {sp:>8.2f}")
        
        # رسم نمودار مقایسه
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, sp_scores)
        plt.title("SP Score Comparison Across Methods")
        plt.ylabel("SP Score")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # افزودن مقادیر روی میله‌ها
        for bar, sp in zip(bars, sp_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{sp:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # ذخیره نمودار
        plot_path = os.path.join(exp_dir, "results", "comparison_plot.png")
        plt.savefig(plot_path, dpi=300)
        print(f"\nComparison plot saved to: {plot_path}")
        
        plt.show()
    
    # تحلیل پارامترهای بهینه
    print(f"\nParameter Analysis:")
    params = stats['optimal_params']
    
    # مقایسه با مقادیر پیش‌فرض
    default_params = {
        'pop_size': 50,
        'max_iter': 100,
        'low': -5.0,
        'high': 5.0,
        'eq_pool_size': 5,
        'p_mut': 0.15,
        'p_reseed': 0.02,
        'levy_scale': 0.02,
        'a1': 2.0,
        'a2': 1.0
    }
    
    for param in params:
        if param in default_params:
            default = default_params[param]
            optimal = params[param]
            change = ((optimal - default) / default * 100) if default != 0 else 0
            print(f"  {param:<15}: Default={default:>6.3f}, Optimal={optimal:>6.3f}, Change={change:>6.1f}%")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <experiment_directory>")
        sys.exit(1)
    
    exp_dir = sys.argv[1]
    analyze_experiment(exp_dir)