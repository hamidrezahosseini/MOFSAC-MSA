# aligners/run_mafft.py
import subprocess
import os

def run_mafft(input_fasta, mafft_path, output_fasta):
    """
    Call mafft (mafft.bat) and redirect stdout to output_fasta.
    mafft_path: full path to mafft.bat
    """
    cmd = [mafft_path, "--auto", input_fasta]
    with open(output_fasta, "w") as out:
        proc = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"MAFFT failed: {proc.stderr}")
    return output_fasta
