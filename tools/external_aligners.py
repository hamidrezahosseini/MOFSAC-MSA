# tools/external_aligners.py

import subprocess
import tempfile
import os

from .msa_scoring import sp_score


# -----------------------------
#  Run MAFFT
# -----------------------------
def run_mafft(sequences, mafft_path=r"E:\PhD\1. first semester\Dr Mansouri\AE_DQN\Code_V3\Classic\mafft-win\mafft.bat"):
    """
    Runs MAFFT externally and returns alignment as list of strings.
    """

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "in.fasta")
        out_path = os.path.join(tmp, "out.fasta")

        # write input FASTA
        with open(in_path, "w") as f:
            for i, s in enumerate(sequences):
                f.write(f">seq{i}\n{s}\n")

        cmd = [mafft_path, "--auto", in_path]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = p.communicate()

        # save to file
        with open(out_path, "w") as f:
            f.write(out)

        # read alignment
        return read_fasta_small(out_path)


# -----------------------------
#  Run ClustalW
# -----------------------------
def run_clustalw(sequences, clustal_path=r"C:\Program Files (x86)\ClustalW2\clustalw2.exe"):
    """
    ClustalW returns an .aln file which we convert to FASTA.
    """

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "in.fasta")
        aln_path = os.path.join(tmp, "in.aln")

        # write FASTA
        with open(in_path, "w") as f:
            for i, s in enumerate(sequences):
                f.write(f">seq{i}\n{s}\n")

        cmd = [clustal_path, "-INFILE=" + in_path]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        return parse_clustal_aln(aln_path)


# -----------------------------
# FASTA parser
# -----------------------------

def read_fasta_small(path):
    seqs = []
    with open(path) as f:
        curr = ""
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if curr:
                    seqs.append(curr)
                curr = ""
            else:
                curr += line
        if curr:
            seqs.append(curr)
    return seqs


# -----------------------------
# Clustal .aln parser
# -----------------------------
def parse_clustal_aln(path, expected_order=None):
    """
    Extremely robust ClustalW .aln parser.
    - Handles multiple blocks
    - Removes annotation lines (* : .)
    - Preserves original FASTA sequence order
    - Ensures all sequences have equal length
    """

    seq_dict = {}
    order = []

    with open(path) as f:
        for line in f:
            line = line.rstrip()

            if not line:
                continue

            # skip header
            if line.startswith("CLUSTAL") or line.startswith("MUSCLE"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            name, fragment = parts[0], parts[1]

            # annotation line (* : .)
            if all(c in "*:." for c in fragment):
                continue

            if name not in seq_dict:
                seq_dict[name] = []
                order.append(name)

            seq_dict[name].append(fragment)

    # Build sequences
    alignment_raw = {name: "".join(seq_dict[name]) for name in seq_dict}

    # If expected ordering provided (i.e. FASTA order), reorder
    if expected_order is not None:
        aligned = []
        for name in expected_order:
            if name not in alignment_raw:
                raise ValueError(f"ClustalW output missing sequence: {name}")
            aligned.append(alignment_raw[name])
    else:
        aligned = [alignment_raw[name] for name in order]

    # Make all sequences equal-length
    max_len = max(len(s) for s in aligned)
    final = []

    for seq in aligned:
        if len(seq) < max_len:
            seq = seq + "-" * (max_len - len(seq))
        final.append(seq)

    return final
