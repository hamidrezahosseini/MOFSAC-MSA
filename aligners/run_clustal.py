# aligners/run_clustal.py
import subprocess
import os

def run_clustal(input_fasta, clustal_path, output_aln):
    """
    Call ClustalW (clustalw2.exe or clustalw.exe). It typically writes .aln file next to input.
    Provide output_aln path where you expect the .aln to be located afterward.
    """
    # ClustalW uses -INFILE=... -OUTFILE=...
    cmd = [clustal_path, f"-INFILE={input_fasta}", f"-OUTFILE={output_aln}", "-TYPE=DNA", "-ALIGN"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ClustalW failed: {proc.stderr}")
    return output_aln
