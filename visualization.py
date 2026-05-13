import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

class AlignmentVisualizer:
    """کلاس برای مصورسازی همترازی‌ها و نتایج مقایسه"""
    
    @staticmethod
    def create_comparison_dataframe(all_metrics: Dict[str, Dict]) -> pd.DataFrame:
        """ایجاد DataFrame از نتایج تمام روش‌ها"""
        rows = []
        for method_name, metrics in all_metrics.items():
            row = {'Method': method_name}
            row.update(metrics)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    @staticmethod
    def plot_comprehensive_comparison(all_metrics: Dict[str, Dict], 
                                     save_path: str = None,
                                     show_plot: bool = True):
        """نمودار جامع مقایسه تمام معیارها"""
        
        df = AlignmentVisualizer.create_comparison_dataframe(all_metrics)
        
        # معیارهای اصلی برای نمایش
        primary_metrics = [
            'sp_score', 'tc_score', 'cs_score', 
            'conservation_score', 'gap_percentage', 
            'execution_time', 'avg_identity'
        ]
        
        metric_names = {
            'sp_score': 'SP Score',
            'tc_score': 'TC Score',
            'cs_score': 'CS Score',
            'conservation_score': 'Conservation',
            'gap_percentage': 'Gap %',
            'execution_time': 'Time (s)',
            'avg_identity': 'Avg Identity %',
            'shannon_entropy': 'Entropy',
            'alignment_length': 'Length',
            'compression_ratio': 'Compression'
        }
        
        # ایجاد subplot‌ها
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()
        
        # رسم هر معیار
        for idx, metric in enumerate(primary_metrics[:9]):
            ax = axes[idx]
            
            # مرتب‌سازی بر اساس مقدار
            sorted_df = df.sort_values(metric, ascending=False)
            
            # تعیین رنگ‌ها
            colors = []
            for i, row in sorted_df.iterrows():
                if row['Method'] == 'EO-Pro (RL)':
                    colors.append('#FF6B6B')  # قرمز برای روش ما
                elif row['Method'] == 'Final_Refined':
                    colors.append('#4ECDC4')  # فیروزه‌ای برای روش نهایی
                elif 'MAFFT' in row['Method']:
                    colors.append('#45B7D1')  # آبی
                elif 'Clustal' in row['Method']:
                    colors.append('#96CEB4')  # سبز
                else:
                    colors.append('#FFEAA7')  # زرد
            
            bars = ax.barh(sorted_df['Method'], sorted_df[metric], color=colors)
            ax.set_xlabel(metric_names.get(metric, metric))
            ax.set_title(f'{metric_names.get(metric, metric)} Comparison')
            ax.grid(axis='x', alpha=0.3)
            
            # افزودن مقادیر روی میله‌ها
            for bar, value in zip(bars, sorted_df[metric]):
                if metric == 'execution_time':
                    text = f'{value:.2f}s'
                elif metric in ['sp_score', 'tc_score', 'cs_score', 'conservation_score']:
                    text = f'{value:.3f}'
                elif metric == 'gap_percentage':
                    text = f'{value:.1f}%'
                elif metric == 'avg_identity':
                    text = f'{value:.1f}%'
                else:
                    text = f'{value:.2f}'
                
                width = bar.get_width()
                ax.text(width + max(sorted_df[metric]) * 0.01, 
                       bar.get_y() + bar.get_height()/2,
                       text, va='center', fontsize=9)
        
        # حذف axes اضافی
        for idx in range(len(primary_metrics[:9]), 9):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Comprehensive Alignment Method Comparison', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    @staticmethod
    def plot_radar_chart(all_metrics: Dict[str, Dict], 
                        save_path: str = None,
                        show_plot: bool = True):
        """نمودار راداری برای مقایسه چندبعدی"""
        
        # انتخاب 4 روش برتر بر اساس SP Score
        df = AlignmentVisualizer.create_comparison_dataframe(all_metrics)
        top_methods = df.nlargest(4, 'sp_score')['Method'].tolist()
        
        # انتخاب معیارها برای رادار
        radar_metrics = ['sp_score', 'tc_score', 'cs_score', 
                        'conservation_score', 'avg_identity']
        
        # نرمال‌سازی مقادیر (0-1)
        normalized_data = {}
        for method in top_methods:
            method_metrics = all_metrics[method]
            normalized = {}
            for metric in radar_metrics:
                value = method_metrics[metric]
                # هر معیار محدوده متفاوتی دارد، نرمال‌سازی ساده
                if metric == 'gap_percentage':
                    # برای gap درصد کمتر بهتر است، بنابراین معکوس می‌کنیم
                    normalized[metric] = 1 - min(value / 100, 1)
                elif metric == 'execution_time':
                    # زمان کمتر بهتر است
                    max_time = max([m['execution_time'] for m in all_metrics.values()])
                    normalized[metric] = 1 - (value / max_time if max_time > 0 else 0)
                else:
                    # برای بقیه، مقدار اصلی (فرض می‌کنیم بیشتر بهتر است)
                    max_val = max([m[metric] for m in all_metrics.values()])
                    normalized[metric] = value / max_val if max_val > 0 else 0
            
            normalized_data[method] = normalized
        
        # ایجاد نمودار رادار
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='radar'))
        
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]  # بسته کردن دایره
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for idx, (method, data) in enumerate(normalized_data.items()):
            values = [data[metric] for metric in radar_metrics]
            values += values[:1]  # بسته کردن
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric.replace('_', ' ').title() for metric in radar_metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Radar Chart Comparison of Top 4 Methods', size=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path.replace('.png', '_radar.png'), dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    @staticmethod
    def plot_tradeoff_analysis(all_metrics: Dict[str, Dict],
                              save_path: str = None,
                              show_plot: bool = True):
        """تحلیل trade-off بین دقت و زمان"""
        
        df = AlignmentVisualizer.create_comparison_dataframe(all_metrics)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        colors = []
        sizes = []
        for i, row in df.iterrows():
            if row['Method'] == 'EO-Pro (RL)':
                colors.append('#FF6B6B')
                sizes.append(300)
            elif row['Method'] == 'Final_Refined':
                colors.append('#4ECDC4')
                sizes.append(300)
            elif 'MAFFT' in row['Method']:
                colors.append('#45B7D1')
                sizes.append(200)
            elif 'Clustal' in row['Method']:
                colors.append('#96CEB4')
                sizes.append(200)
            else:
                colors.append('#FFEAA7')
                sizes.append(150)
        
        scatter = ax.scatter(df['execution_time'], df['sp_score'], 
                           s=sizes, c=colors, alpha=0.8, edgecolors='black')
        
        ax.set_xlabel('Execution Time (seconds)', fontsize=12)
        ax.set_ylabel('SP Score', fontsize=12)
        ax.set_title('Accuracy vs. Runtime Trade-off Analysis', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # افزودن نام روش‌ها
        for i, row in df.iterrows():
            ax.annotate(row['Method'], 
                       (row['execution_time'], row['sp_score']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, alpha=0.8)
        
        # محاسبه و نمایش Pareto frontier
        try:
            from scipy.spatial import ConvexHull
            points = df[['execution_time', 'sp_score']].values
            hull = ConvexHull(points)
            
            # نقاط روی مرز
            frontier_points = points[hull.vertices]
            frontier_points = frontier_points[frontier_points[:, 0].argsort()]
            
            ax.plot(frontier_points[:, 0], frontier_points[:, 1], 
                   'r--', alpha=0.7, linewidth=2, label='Pareto Frontier')
            ax.legend()
        except:
            pass
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path.replace('.png', '_tradeoff.png'), dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    @staticmethod
    def generate_html_report(all_metrics: Dict[str, Dict], 
                           save_path: str = "comparison_report.html"):
        """ایجاد گزارش HTML تعاملی"""
        
        df = AlignmentVisualizer.create_comparison_dataframe(all_metrics)
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>RNA Alignment Method Comparison Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                h2 { color: #555; margin-top: 30px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: center; }
                th { background-color: #4ECDC4; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .best { background-color: #d4edda !important; font-weight: bold; }
                .good { background-color: #fff3cd !important; }
                .metric-header { background-color: #45B7D1 !important; color: white; }
            </style>
        </head>
        <body>
            <h1>RNA Sequence Alignment Method Comparison Report</h1>
            <p>Generated on: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            
            <h2>1. Summary Table</h2>
            <table>
                <tr>
                    <th>Method</th>
                    <th>SP Score</th>
                    <th>TC Score</th>
                    <th>CS Score</th>
                    <th>Conservation</th>
                    <th>Gap %</th>
                    <th>Time (s)</th>
                    <th>Avg Identity</th>
                </tr>
        """
        
        # پیدا کردن بهترین مقادیر برای هر ستون
        best_values = {}
        for metric in ['sp_score', 'tc_score', 'cs_score', 'conservation_score', 'avg_identity']:
            best_values[metric] = df[metric].max()
        best_values['gap_percentage'] = df['gap_percentage'].min()
        best_values['execution_time'] = df['execution_time'].min()
        
        # اضافه کردن سطرهای جدول
        for _, row in df.iterrows():
            html_content += f"""
                <tr>
                    <td><strong>{row['Method']}</strong></td>
            """
            
            # برای هر معیار، تعیین کلاس CSS
            for metric in ['sp_score', 'tc_score', 'cs_score', 'conservation_score', 
                          'gap_percentage', 'execution_time', 'avg_identity']:
                value = row[metric]
                best = best_values[metric]
                
                if metric == 'gap_percentage' or metric == 'execution_time':
                    # مقادیر کمتر بهتر هستند
                    is_best = abs(value - best) < 0.0001
                else:
                    # مقادیر بیشتر بهتر هستند
                    is_best = abs(value - best) < 0.0001
                
                cell_class = "best" if is_best else ""
                
                if metric == 'execution_time':
                    display_value = f"{value:.2f}s"
                elif metric == 'gap_percentage':
                    display_value = f"{value:.1f}%"
                elif metric == 'avg_identity':
                    display_value = f"{value:.1f}%"
                else:
                    display_value = f"{value:.3f}"
                
                html_content += f'<td class="{cell_class}">{display_value}</td>'
            
            html_content += "</tr>"
        
        html_content += """
            </table>
            
            <h2>2. Key Findings</h2>
            <ul>
                <li><strong>Best SP Score:</strong> """ + df.loc[df['sp_score'].idxmax(), 'Method'] + """ (""" + f"{df['sp_score'].max():.2f}" + """)</li>
                <li><strong>Best TC Score:</strong> """ + df.loc[df['tc_score'].idxmax(), 'Method'] + """ (""" + f"{df['tc_score'].max():.3f}" + """)</li>
                <li><strong>Fastest Method:</strong> """ + df.loc[df['execution_time'].idxmin(), 'Method'] + """ (""" + f"{df['execution_time'].min():.2f}s" + """)</li>
                <li><strong>Most Compact Alignment:</strong> """ + df.loc[df['gap_percentage'].idxmin(), 'Method'] + """ (""" + f"{df['gap_percentage'].min():.1f}%" + """)</li>
            </ul>
            
            <h2>3. Recommendations</h2>
            <p>Based on the analysis:</p>
            <ul>
                <li><strong>For maximum accuracy:</strong> Use """ + df.loc[df['sp_score'].idxmax(), 'Method'] + """</li>
                <li><strong>For speed-critical applications:</strong> Use """ + df.loc[df['execution_time'].idxmin(), 'Method'] + """</li>
                <li><strong>For balanced performance:</strong> Consider methods on the Pareto frontier</li>
            </ul>
            
            <h2>4. Detailed Metrics</h2>
            <table>
                <tr class="metric-header">
                    <th>Metric</th>
                    <th>Description</th>
                    <th>Ideal Value</th>
                </tr>
                <tr>
                    <td><strong>SP Score</strong></td>
                    <td>Sum-of-Pairs score: Higher is better</td>
                    <td>Maximize</td>
                </tr>
                <tr>
                    <td><strong>TC Score</strong></td>
                    <td>Total Column score: % perfect columns</td>
                    <td>Maximize</td>
                </tr>
                <tr>
                    <td><strong>CS Score</strong></td>
                    <td>Column Score: % high-conservation columns</td>
                    <td>Maximize</td>
                </tr>
                <tr>
                    <td><strong>Conservation</strong></td>
                    <td>Average similarity per column (0-1)</td>
                    <td>Maximize</td>
                </tr>
                <tr>
                    <td><strong>Gap %</strong></td>
                    <td>Percentage of gap characters</td>
                    <td>Minimize</td>
                </tr>
                <tr>
                    <td><strong>Time (s)</strong></td>
                    <td>Execution time in seconds</td>
                    <td>Minimize</td>
                </tr>
                <tr>
                    <td><strong>Avg Identity</strong></td>
                    <td>Average pairwise identity (%)</td>
                    <td>Maximize</td>
                </tr>
            </table>
        </body>
        </html>
        """
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML report saved to: {save_path}")
        
        return save_path