# tools/local_refinement.py

from copy import deepcopy
from .msa_scoring import sp_score

def remove_empty_cols(aln):
    L = len(aln[0])
    keep = []
    for c in range(L):
        col = [seq[c] for seq in aln]
        if not all(ch == '-' for ch in col):
            keep.append(c)

    new = []
    for seq in aln:
        new.append("".join(seq[c] for c in keep))
    return new

def local_refine_alignment(aln, max_iters=30):
    """
    Minimal local refinement:
    - remove empty columns
    - iteratively remove columns with >80% gaps
    """

    best = deepcopy(aln)
    best_score = sp_score(best)

    for _ in range(max_iters):
        new = remove_empty_cols(best)

        # remove columns with too many gaps
        L = len(new[0])
        keep = []
        for c in range(L):
            col = [s[c] for s in new]
            gap_fraction = col.count('-') / len(col)
            if gap_fraction < 0.80:
                keep.append(c)

        refined = []
        for s in new:
            refined.append("".join(s[c] for c in keep))

        score = sp_score(refined)
        if score <= best_score:
            break

        best = refined
        best_score = score

    return best
