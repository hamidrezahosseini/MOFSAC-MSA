# utils/fasta_io.py
def read_fasta(filepath):
    names = []
    seqs = []
    with open(filepath, 'r') as f:
        name = None
        seq_lines = []
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    names.append(name)
                    seqs.append(''.join(seq_lines))
                name = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        if name is not None:
            names.append(name)
            seqs.append(''.join(seq_lines))
    return names, seqs

def write_fasta(names, seqs, outfile):
    with open(outfile, 'w') as f:
        for n, s in zip(names, seqs):
            f.write(f">{n}\n")
            # wrap lines at 80
            for i in range(0, len(s), 80):
                f.write(s[i:i+80] + "\n")
