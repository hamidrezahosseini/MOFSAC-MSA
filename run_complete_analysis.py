"""
اسکریپت اجرای کامل آنالیز و مقایسه
"""
import os
import sys
import json
from datetime import datetime

# اضافه کردن مسیر پروژه
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from final_pipeline import run_final_pipeline
from visualization import AlignmentVisualizer

def run_complete_analysis(fasta_path="input.fasta"):
    """اجرای کامل آنالیز و تولید گزارش‌های جامع"""
    
    print("="*80)
    print("COMPLETE RNA ALIGNMENT ANALYSIS PIPELINE")
    print("="*80)
    
    # اجرای پایپ‌لاین اصلی
    final_alignment, all_metrics = run_final_pipeline(fasta_path)
    
    # ایجاد گزارش‌های اضافی
    print("\n" + "="*80)
    print("GENERATING ADDITIONAL REPORTS")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"reports/analysis_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    
    # ایجاد visualizer
    visualizer = AlignmentVisualizer()
    
    # 1. نمودار جامع
    print("📊 Generating comprehensive comparison charts...")
    visualizer.plot_comprehensive_comparison(
        all_metrics,
        save_path=os.path.join(report_dir, "comprehensive_charts.png"),
        show_plot=True
    )
    
    # 2. نمودار رادار
    print("🎯 Generating radar comparison...")
    visualizer.plot_radar_chart(
        all_metrics,
        save_path=os.path.join(report_dir, "radar_comparison.png"),
        show_plot=True
    )
    
    # 3. تحلیل trade-off
    print("⚖️  Generating accuracy-time trade-off analysis...")
    visualizer.plot_tradeoff_analysis(
        all_metrics,
        save_path=os.path.join(report_dir, "tradeoff_analysis.png"),
        show_plot=True
    )
    
    # 4. گزارش HTML
    print("📄 Generating HTML report...")
    html_path = os.path.join(report_dir, "complete_report.html")
    visualizer.generate_html_report(all_metrics, save_path=html_path)
    
    # 5. ذخیره داده‌ها
    print("💾 Saving all data...")
    data_path = os.path.join(report_dir, "all_metrics.json")
    with open(data_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # 6. خلاصه اجرا
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*80)
    
    # پیدا کردن بهترین روش
    best_sp_method = max(all_metrics.items(), key=lambda x: x[1]['sp_score'])[0]
    best_sp_value = max([m['sp_score'] for m in all_metrics.values()])
    
    best_time_method = min(all_metrics.items(), key=lambda x: x[1]['execution_time'])[0]
    best_time_value = min([m['execution_time'] for m in all_metrics.values()])
    
    print(f"\n🏆 **Best SP Score:** {best_sp_method} ({best_sp_value:.0f})")
    print(f"⚡ **Fastest Method:** {best_time_method} ({best_time_value:.2f}s)")
    print(f"📁 **Report Directory:** {report_dir}")
    print(f"📊 **HTML Report:** {html_path}")
    print(f"📈 **Charts:** {report_dir}/")
    
    # پیشنهادات
    print("\n💡 **Recommendations:**")
    if best_sp_method != best_time_method:
        print(f"  • For accuracy: Use {best_sp_method}")
        print(f"  • For speed: Use {best_time_method}")
    else:
        print(f"  • {best_sp_method} is both accurate and fast!")
    
    return all_metrics, report_dir

if __name__ == "__main__":
    # بررسی وجود فایل ورودی
    if not os.path.exists("input.fasta"):
        print("Creating sample input file...")
        sample_seqs = [
            "AUCGUAUCGUAUCGUAUCGUAUCGU",
            "AUCGUAUCG-AUCGUAUCGUAUCGU",
            "AUCG-AUCGUAUCGUAUCGUAUCGU",
            "AUCGUAUCGUAUCG-AUCGUAUCGU",
            "AUCGUAUCGUAUCGUAUCG-AUCGU",
            "AUCGUAUCGUAUCGUAUCGUAUCG-",
            "-UCGUAUCGUAUCGUAUCGUAUCGU",
            "AUCGUAUCGUAUCGUAUCGUAUCGA"
        ]
        
        with open("input.fasta", "w") as f:
            for i, seq in enumerate(sample_seqs):
                f.write(f">seq_{i+1}\n{seq}\n")
        print("Sample file created.")
    
    # اجرای آنالیز کامل
    try:
        metrics, report_dir = run_complete_analysis("input.fasta")
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 All results saved in: {report_dir}")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()