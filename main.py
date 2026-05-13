#main.py
import os
import time
from copy import deepcopy

# plotting
import matplotlib.pyplot as plt

# -------------------------------------------------------
#                   Internal Tools
# -------------------------------------------------------
from tools.msa_scoring import sp_score
from tools.local_refinement import local_refine_alignment
from tools.external_aligners import run_mafft, run_clustalw

# Hybrid-Pro Representation
from representation.hybrid_pro import build_hybrid_spec, decode_hybrid

# EO-Pro Optimizer
from eo.equilibrium_optimizer_pro import equilibrium_optimizer_pro


# -------------------------------------------------------
#                 FASTA Reader / Writer
# -------------------------------------------------------
def read_fasta(path):
    seqs = []
    with open(path) as f:
        name = None
        seq = ""
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    seqs.append(seq)
                name = line[1:]
                seq = ""
            else:
                seq += line
        if seq:
            seqs.append(seq)
    return seqs


def write_fasta(path, alignment):
    with open(path, "w") as f:
        for i, seq in enumerate(alignment):
            f.write(f">seq{i}\n{seq}\n")


# -------------------------------------------------------
#                   Fuzzy Fitness Wrapper
# -------------------------------------------------------
from fuzzy.fuzzy_full import fuzzy_total_membership

def fuzzy_fitness(alignment):
    """
    EO-Pro fitness = fuzzy multi-objective score
    """
    # محاسبه امتیاز فازی
    mu_total, details = fuzzy_total_membership(alignment)
    
    # وزن‌دهی به امتیاز فازی
    fuzzy_score = mu_total * 1000000  # مقیاس‌سازی
    
    # همچنین SP Score را هم در نظر بگیریم
    sp = sp_score(alignment)
    
    # ترکیب دو امتیاز (وزن بیشتر به فازی)
    combined_score = (0.7 * fuzzy_score) + (0.3 * sp)
    
    return combined_score, {"sp": sp, "fuzzy": mu_total, "details": details}


# -------------------------------------------------------
#                      Helpers
# -------------------------------------------------------
def save_alignment_files(out_dir, aln_dict):
    """
    aln_dict: {"name": alignment(list[str]), ...}
    saves each alignment to a fasta file named <name>.fasta inside out_dir
    """
    os.makedirs(out_dir, exist_ok=True)
    for name, aln in aln_dict.items():
        path = os.path.join(out_dir, f"{name}.fasta")
        write_fasta(path, aln)


def print_alignment(title, alignment):
    print(f"\n=== {title} ===")
    for s in alignment:
        print(s)


# -------------------------------------------------------
#                       PIPELINE
# -------------------------------------------------------
def pipeline():

    # -----------------------------------------
    # 1. Load Input Sequences
    # -----------------------------------------
    fasta_path = "input.fasta"
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"Input FASTA not found: {fasta_path}")
    sequences = read_fasta(fasta_path)
    print(f"Loaded {len(sequences)} sequences.")

    # -----------------------------------------
    # 2. Run MAFFT + ClustalW Baselines
    # -----------------------------------------
    print("Running MAFFT...")
    aln_mafft = run_mafft(sequences)
    sp_mafft = sp_score(aln_mafft)
    print(f"SP(MAFFT): {sp_mafft}")

    print("Running ClustalW...")
    aln_clustal = run_clustalw(sequences)
    sp_clustal = sp_score(aln_clustal)
    print(f"SP(CLUSTAL): {sp_clustal}")

    # -----------------------------------------
    # 3. Seed Alignment Selection
    # -----------------------------------------
    if sp_clustal >= sp_mafft:
        print("Using seed alignment: ClustalW")
        seed = deepcopy(aln_clustal)
        seed_name = "clustalw"
    else:
        print("Using seed alignment: MAFFT")
        seed = deepcopy(aln_mafft)
        seed_name = "mafft"

    # Count gaps
    total_gaps = sum(s.count('-') for s in seed)
    print(f"Total gaps in seed: {total_gaps}")

    # show seed briefly
    print_alignment(f"Selected seed ({seed_name})", seed)

    # -----------------------------------------
    # 4. Build Hybrid-Pro Representation
    # -----------------------------------------
    print("Building Hybrid-Pro representation...")
    spec = build_hybrid_spec(
        seed,
        K_insert=8,
        K_delete=8,
        M_swaps=6,
        S_segments=8
    )

    print(f"Hybrid-Pro dimension: {spec['dim']}")

    # EO-Pro requires decode_fn(v, spec, max_shift)
    def decode_wrapper(v, spec_in, max_shift=20):
        return decode_hybrid(seed, v, spec, max_shift=max_shift)

    # -----------------------------------------
    # 5. Run EO-Pro
    # -----------------------------------------
    print("Running Equilibrium Optimizer (EO-Pro)...")

    eo_res = equilibrium_optimizer_pro(
        decode_fn=decode_wrapper,
        fitness_fn=fuzzy_fitness,
        spec=spec,
        pop_size=50,
        max_iter=100,
        low=-5.0,
        high=5.0,
        eq_pool_size=5,
        p_mut=0.15,
        p_reseed=0.02,
        levy_scale=0.02,
        seed=None,
        verbose=True,
        max_eval=None
    )

    best_alignment = eo_res["best_alignment"]
    best_fit = eo_res["best_fitness"]

    # compute raw SP of EO best (before refinement)
    sp_eo_raw = sp_score(best_alignment)

    print(f"\nEO best fuzzy fitness: {best_fit}  (SP of this alignment: {sp_eo_raw})")

    # print EO best alignment (before refinement)
    print_alignment("EO Best Alignment (before refinement)", best_alignment)

    # -----------------------------------------
    # 6. Local Refinement
    # -----------------------------------------
    print("Running local refinement...")
    refined = local_refine_alignment(best_alignment, max_iters=50)
    # tools.local_refinement.local_refine_alignment returns refined alignment (list[str])
    refined_sp = sp_score(refined)

    print(f"Refined SP score: {refined_sp}")

    print_alignment("EO Refined Alignment", refined)

    # -----------------------------------------
    # 7. Save Results (alignments + metadata)
    # -----------------------------------------
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    aln_outputs = {
        "mafft": aln_mafft,
        "clustalw": aln_clustal,
        "eo_best_before_refine": best_alignment,
        "eo_refined": refined
    }
    save_alignment_files(out_dir, aln_outputs)
    print(f"\nSaved alignments to folder: {out_dir}")

    # also save a small text summary
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("SP scores summary\n")
        f.write("===================\n")
        f.write(f"SP(MAFFT): {sp_mafft}\n")
        f.write(f"SP(CLUSTAL): {sp_clustal}\n")
        f.write(f"SP(EO_raw): {sp_eo_raw}\n")
        f.write(f"SP(EO_refined): {refined_sp}\n")
        f.write("\nNote: EO fitness function used SP as objective (fuzzy wrapper returns SP in details).\n")
    print(f"Saved summary to: {summary_path}")

    # -----------------------------------------
    # 8. Plot Comparison
    # -----------------------------------------
    methods = ["MAFFT", "ClustalW", "EO (before refine)", "EO (refined)"]
    scores = [sp_mafft, sp_clustal, sp_eo_raw, refined_sp]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, scores)
    plt.title("SP Score Comparison")
    plt.ylabel("SP Score")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    # annotate bars with values
    for rect, val in zip(bars, scores):
        height = rect.get_height()
        plt.annotate(f"{int(val)}", xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    plt.tight_layout()

    plot_path = os.path.join(out_dir, "sp_comparison.png")
    plt.savefig(plot_path, dpi=200)
    print(f"Saved SP comparison plot to: {plot_path}")

    try:
        plt.show()
    except Exception:
        # in headless environments plt.show may fail; we already saved the figure
        pass

    print(f"\nFinal SP (refined): {refined_sp}")

    # -----------------------------------------
    # 9. Done
    # -----------------------------------------
    print("\nPipeline finished.")


# -------------------------------------------------------
#                       MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    pipeline()
