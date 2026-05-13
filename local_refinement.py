# local_refinement.py
from copy import deepcopy

def local_refinement(alignment, max_iter=10):
    """
    جابه‌جایی gap ها برای بهبود ستون‌ها
    """
    alignment = [list(s) for s in alignment]

    for _ in range(max_iter):
        for i in range(len(alignment)):
            for j in range(1, len(alignment[i]) - 1):
                if alignment[i][j] == "-" and alignment[i][j-1] != "-":
                    # shift gap left
                    alignment[i][j], alignment[i][j-1] = alignment[i][j-1], "-"

    return ["".join(s) for s in alignment]


def _remove_full_gap_columns(aln):
    """
    ستون‌هایی که در همه سکانس‌ها گپ هستند حذف می‌شوند
    """
    if not aln:
        return aln

    num_seqs = len(aln)
    length = len(aln[0])

    keep_cols = []
    for i in range(length):
        col = [aln[s][i] for s in range(num_seqs)]
        if not all(c == '-' for c in col):
            keep_cols.append(i)

    new_aln = []
    for s in range(num_seqs):
        new_seq = "".join(aln[s][i] for i in keep_cols)
        new_aln.append(new_seq)

    return new_aln
