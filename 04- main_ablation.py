"""
Ablation Study for RNA MSA Pipeline
Seed is ALWAYS selected from MAFFT / ClustalW (best SP)
Processes ALL fasta files in ablation_input/
"""

import os

from utils.fasta_io import read_fasta, write_fasta
from tools.external_aligners import run_mafft, run_clustalw
from tools.msa_scoring import (
    sp_score,
    cs_score,
    gap_ratio,
    alignment_length,
)
from tools.local_refinement import local_refine_alignment

from simple_representation import DisplacementDecoder
from advanced_fitness import AdvancedFitness
from optimized_eo import OptimizedEO

# ✅ FULL PIPELINE (WITH SAC INSIDE)
from main_run_with_rl import run_final_pipeline_improved


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def load_sequences(fasta_path):
    headers, seqs = read_fasta(fasta_path)
    return headers, seqs


def evaluate(alignment):
    """
    Evaluates an alignment using standard MSA metrics.
    Returns a dictionary compatible with ablation reporting.
    """
    return {
        "SP": sp_score(alignment),
        "CS": cs_score(alignment),
        "GAP": gap_ratio(alignment),
        "Len": alignment_length(alignment),
    }


def select_seed(sequences):
    """
    Seed is selected strictly between MAFFT and ClustalW
    using SP score (exact old behavior).
    """
    aln_mafft = run_mafft(sequences)
    aln_clustal = run_clustalw(sequences)

    sp_mafft = sp_score(aln_mafft)
    sp_clustal = sp_score(aln_clustal)

    if sp_clustal >= sp_mafft:
        return aln_clustal, "ClustalW", aln_mafft
    else:
        return aln_mafft, "MAFFT", aln_clustal


# --------------------------------------------------
# Core runner
# --------------------------------------------------

def run_method(method_name, seed_alignment, fasta_path=None):
    """
    Executes a single ablation method.

    NOTE:
    - EO-based methods: NO RL
    - Full_Method: includes SAC + multi-stage refinement
    """

    # ---------------- Seed only ----------------
    if method_name == "SeedOnly":
        return seed_alignment

    # ---------------- FULL PIPELINE ----------------
    if method_name == "Full_Method":
        print("🚀 Running FULL improved pipeline (with SAC)")
        final_alignment, _, _, _ = run_final_pipeline_improved(fasta_path)
        return final_alignment

    # ---------------- EO-based methods (NO RL) ----------------
    representation = DisplacementDecoder(seed_alignment)
    dim = representation.dim

    def decode_fn(vec):
        return representation.decode(vec)

    # ---------- Fitness ----------
    if method_name == "EO_SPOnly":
        def fitness_fn(aln):
            return sp_score(aln), {}
    else:
        fitness_calc = AdvancedFitness()

        def fitness_fn(aln):
            return fitness_calc.calculate(aln)

    pop_size = min(50, max(20, dim * 2))

    eo = OptimizedEO(
        decode_fn=decode_fn,
        fitness_fn=fitness_fn,
        dim=dim,
        pop_size=pop_size,
        max_iter=30,  # ✅ reduced iterations
        use_equilibrium_pool=(method_name != "EO_SPOnly"),
        seed=42,
    )

    # ✅ local search only (prevent collapse)
    result = eo.optimize(low=-0.15, high=0.15)
    alignment = result["best_alignment"]

    # ✅ elitism: never worse than seed
    if sp_score(alignment) < sp_score(seed_alignment):
        alignment = seed_alignment

    # ✅ refinement only if EO improved or matched seed
    if method_name in ["EO_AdvancedFitness", "EO_GapAware"]:
        if sp_score(alignment) >= sp_score(seed_alignment):
            alignment = local_refine_alignment(alignment, max_iters=40)

    return alignment


# --------------------------------------------------
# Main Ablation Loop
# --------------------------------------------------

def main():
    input_dir = "ablation_input"
    output_root = "ablation_results"
    os.makedirs(output_root, exist_ok=True)

    fasta_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".fasta", ".fa"))
    ]

    methods = [
        "SeedOnly",
        "EO_SPOnly",
        "EO_AdvancedFitness",
        "EO_GapAware",
        "Full_Method",   # ✅ ONLY METHOD WITH SAC
    ]

    for fasta_file in fasta_files:
        print(f"\n🔬 Processing {fasta_file}")

        base_name = os.path.splitext(fasta_file)[0]
        out_dir = os.path.join(output_root, base_name)
        os.makedirs(out_dir, exist_ok=True)

        fasta_path = os.path.join(input_dir, fasta_file)

        # ---------- Load ----------
        headers, sequences = load_sequences(fasta_path)

        # ---------- Seed ----------
        seed, seed_name, other_baseline = select_seed(sequences)

        baselines = {
            "MAFFT": seed if seed_name == "MAFFT" else other_baseline,
            "ClustalW": seed if seed_name == "ClustalW" else other_baseline,
        }

        summary_path = os.path.join(out_dir, "summary.txt")

        with open(summary_path, "w", encoding="utf-8") as f:
            # ✅ Updated header
            f.write("Method\tSeed\tSP\tCS\tGap(%)\tLen\n")

            # ---------- Baselines ----------
            for name, aln in baselines.items():
                metrics = evaluate(aln)
                write_fasta(headers, aln, os.path.join(out_dir, f"{name}.fasta"))
                f.write(
                    f"{name}\t-\t"
                    f"{metrics['SP']:.0f}\t"
                    f"{metrics['CS']:.4f}\t"
                    f"{metrics['GAP']:.2f}\t"
                    f"{metrics['Len']}\n"
                )

            # ---------- Ablation ----------
            for m in methods:
                aln = run_method(m, seed, fasta_path=fasta_path)
                metrics = evaluate(aln)

                write_fasta(headers, aln, os.path.join(out_dir, f"{m}.fasta"))
                f.write(
                    f"{m}\t{seed_name}\t"
                    f"{metrics['SP']:.0f}\t"
                    f"{metrics['CS']:.4f}\t"
                    f"{metrics['GAP']:.2f}\t"
                    f"{metrics['Len']}\n"
                )

        print(f"✅ Finished {fasta_file} → {out_dir}")


if __name__ == "__main__":
    main()
