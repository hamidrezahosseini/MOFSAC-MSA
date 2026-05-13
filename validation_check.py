# validation_check.py
import os
from utils.fasta_io import read_fasta
from tools.msa_scoring import sp_score

def validate_and_compare():
    """بررسی اعتبار نتایج"""
    
    # مسیر نتایج
    results_dir = "results"  # مسیر پوشه results را تغییر دهید اگر متفاوت است
    latest_experiment = sorted([d for d in os.listdir(results_dir) 
                                if d.startswith("final_")], reverse=True)[0]
    
    exp_path = os.path.join(results_dir, latest_experiment)
    print(f"Checking experiment: {exp_path}")
    
    # خواندن همترازی‌های مختلف
    alignments = {}
    for file in os.listdir(exp_path):
        if file.endswith("_alignment.fasta"):
            method = file.replace("_alignment.fasta", "")
            names, seqs = read_fasta(os.path.join(exp_path, file))
            alignments[method] = seqs
    
    # محاسبه SP score برای هر کدام
    print("\n" + "="*60)
    print("VALIDATION CHECK")
    print("="*60)
    
    for method, aln in alignments.items():
        # بررسی طول یکسان
        lengths = [len(seq) for seq in aln]
        if len(set(lengths)) != 1:
            print(f"⚠️  {method}: Sequences have different lengths!")
            continue
            
        # محاسبه SP
        sp = sp_score(aln)
        
        # بررسی کاراکترهای معتبر
        valid_chars = set('ACGTUacgtu-')
        invalid_chars = set()
        for seq in aln:
            for char in seq:
                if char not in valid_chars:
                    invalid_chars.add(char)
        
        if invalid_chars:
            print(f"⚠️  {method}: Invalid characters found: {invalid_chars}")
        else:
            print(f"✓ {method:<15}: SP = {sp:>6}, Length = {len(aln[0])}, Sequences = {len(aln)}")
    
    # محاسبه improvement
    if 'final' in alignments and 'clustal' in alignments:
        final_sp = sp_score(alignments['final'])
        clustal_sp = sp_score(alignments['clustal'])
        improvement = final_sp - clustal_sp
        improvement_percent = (improvement / clustal_sp) * 100
        
        print("\n" + "="*60)
        print("IMPROVEMENT ANALYSIS")
        print("="*60)
        print(f"Final vs ClustalW:")
        print(f"  ClustalW SP: {clustal_sp}")
        print(f"  Final SP:    {final_sp}")
        print(f"  Improvement: {improvement} points ({improvement_percent:.1f}%)")
        
        if improvement > 0:
            print(f"\n✅ VALIDATED: Improvement confirmed!")
        else:
            print(f"\n❌ WARNING: Negative improvement detected!")

if __name__ == "__main__":
    validate_and_compare()