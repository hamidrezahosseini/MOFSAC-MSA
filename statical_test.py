import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon
import warnings
warnings.filterwarnings('ignore')

# خواندن فایل اکسل
file_path = 'staticaltest.xlsx'

# خواندن تمام شیت‌ها
sp_scores = pd.read_excel(file_path, sheet_name='SP_Scores', index_col=0)
cs_scores = pd.read_excel(file_path, sheet_name='CS_Scores', index_col=0)
gap_percentage = pd.read_excel(file_path, sheet_name='Gap_Percentage', index_col=0)

# تبدیل 'NAN' رشته‌ای به NaN عددی
for df in [sp_scores, cs_scores, gap_percentage]:
    df.replace('NAN', np.nan, inplace=True)
    # تبدیل تمام ستون‌ها به عددی
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("=" * 80)
print("تحلیل آماری - SP Scores")
print("=" * 80)

# آمار توصیفی
print("\n--- آمار توصیفی (میانگین ± انحراف معیار) ---")
desc_stats = sp_scores.describe().loc[['mean', 'std', '50%']]
desc_stats.index = ['Mean', 'Std', 'Median']
print(desc_stats.round(2))

# Friedman Test (برای مقایسه کلی تمام روش‌ها)
print("\n--- Friedman Test (مقایسه کلی) ---")
# حذف ردیف‌هایی که NaN دارند
sp_clean = sp_scores.dropna()
if len(sp_clean) >= 3:
    stat, p_value = friedmanchisquare(*[sp_clean[col] for col in sp_clean.columns])
    print(f"Chi-square statistic: {stat:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"نتیجه: {'تفاوت معنی‌دار وجود دارد' if p_value < 0.05 else 'تفاوت معنی‌دار نیست'}")
else:
    print("داده کافی برای Friedman test وجود ندارد")

# مقایسه زوجی: MOFSACEO-MSA با سایر روش‌ها
print("\n--- مقایسه زوجی (Wilcoxon Signed-Rank Test) ---")
print(f"{'Method':<15} {'Mean Diff':<12} {'P-value':<12} {'Effect Size':<12} {'Significant'}")
print("-" * 70)

baseline = 'MOFSACEO-MSA'
results = []

for method in sp_scores.columns:
    if method == baseline:
        continue
    
    # حذف NaN برای هر مقایسه
    paired_data = sp_scores[[baseline, method]].dropna()
    
    if len(paired_data) < 3:
        print(f"{method:<15} {'N/A':<12} {'N/A':<12} {'N/A':<12} Insufficient data")
        continue
    
    # Wilcoxon test
    stat, p_value = wilcoxon(paired_data[baseline], paired_data[method])
    
    # محاسبه Effect Size (Cohen's d)
    diff = paired_data[baseline] - paired_data[method]
    mean_diff = diff.mean()
    std_diff = diff.std()
    effect_size = mean_diff / std_diff if std_diff != 0 else 0
    
    significant = "Yes" if p_value < 0.05 else "No"
    
    print(f"{method:<15} {mean_diff:>11.2f} {p_value:>11.6f} {effect_size:>11.3f} {significant}")
    
    results.append({
        'Metric': 'SP Score',
        'Method': method,
        'Mean_Diff': mean_diff,
        'P_value': p_value,
        'Effect_Size': effect_size,
        'Significant': significant
    })

# تکرار برای CS Scores
print("\n" + "=" * 80)
print("تحلیل آماری - CS Scores")
print("=" * 80)

print("\n--- آمار توصیفی (میانگین ± انحراف معیار) ---")
desc_stats_cs = cs_scores.describe().loc[['mean', 'std', '50%']]
desc_stats_cs.index = ['Mean', 'Std', 'Median']
print(desc_stats_cs.round(4))

print("\n--- مقایسه زوجی (Wilcoxon Signed-Rank Test) ---")
print(f"{'Method':<15} {'Mean Diff':<12} {'P-value':<12} {'Effect Size':<12} {'Significant'}")
print("-" * 70)

for method in cs_scores.columns:
    if method == baseline:
        continue
    
    paired_data = cs_scores[[baseline, method]].dropna()
    
    if len(paired_data) < 3:
        print(f"{method:<15} {'N/A':<12} {'N/A':<12} {'N/A':<12} Insufficient data")
        continue
    
    stat, p_value = wilcoxon(paired_data[baseline], paired_data[method])
    
    diff = paired_data[baseline] - paired_data[method]
    mean_diff = diff.mean()
    std_diff = diff.std()
    effect_size = mean_diff / std_diff if std_diff != 0 else 0
    
    significant = "Yes" if p_value < 0.05 else "No"
    
    print(f"{method:<15} {mean_diff:>11.4f} {p_value:>11.6f} {effect_size:>11.3f} {significant}")
    
    results.append({
        'Metric': 'CS Score',
        'Method': method,
        'Mean_Diff': mean_diff,
        'P_value': p_value,
        'Effect_Size': effect_size,
        'Significant': significant
    })

# تکرار برای Gap Percentage
print("\n" + "=" * 80)
print("تحلیل آماری - Gap Percentage")
print("=" * 80)

print("\n--- آمار توصیفی (میانگین ± انحراف معیار) ---")
desc_stats_gap = gap_percentage.describe().loc[['mean', 'std', '50%']]
desc_stats_gap.index = ['Mean', 'Std', 'Median']
print(desc_stats_gap.round(2))

print("\n--- مقایسه زوجی (Wilcoxon Signed-Rank Test) ---")
print(f"{'Method':<15} {'Mean Diff':<12} {'P-value':<12} {'Effect Size':<12} {'Significant'}")
print("-" * 70)

for method in gap_percentage.columns:
    if method == baseline:
        continue
    
    paired_data = gap_percentage[[baseline, method]].dropna()
    
    if len(paired_data) < 3:
        print(f"{method:<15} {'N/A':<12} {'N/A':<12} {'N/A':<12} Insufficient data")
        continue
    
    stat, p_value = wilcoxon(paired_data[baseline], paired_data[method])
    
    diff = paired_data[baseline] - paired_data[method]
    mean_diff = diff.mean()
    std_diff = diff.std()
    effect_size = mean_diff / std_diff if std_diff != 0 else 0
    
    significant = "Yes" if p_value < 0.05 else "No"
    
    print(f"{method:<15} {mean_diff:>11.2f} {p_value:>11.6f} {effect_size:>11.3f} {significant}")
    
    results.append({
        'Metric': 'Gap Percentage',
        'Method': method,
        'Mean_Diff': mean_diff,
        'P_value': p_value,
        'Effect_Size': effect_size,
        'Significant': significant
    })

# ذخیره نتایج در فایل اکسل
results_df = pd.DataFrame(results)
output_file = 'statistical_results.xlsx'

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='Pairwise_Comparisons', index=False)
    
    # آمار توصیفی
    desc_stats.T.to_excel(writer, sheet_name='SP_Descriptive')
    desc_stats_cs.T.to_excel(writer, sheet_name='CS_Descriptive')
    desc_stats_gap.T.to_excel(writer, sheet_name='Gap_Descriptive')

print("\n" + "=" * 80)
print(f"نتایج در فایل '{output_file}' ذخیره شد.")
print("=" * 80)
