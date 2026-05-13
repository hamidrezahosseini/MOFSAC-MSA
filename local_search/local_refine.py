# local_search/local_refine.py
from scoring.spscore import sp_score
import copy

def local_refine_alignment(alignment, match=2, mismatch=-1, gap=-2, max_iters=100):
    """
    Try simple greedy local moves: for each sequence and each gap, try shift left/right by one
    and accept if SP score improves. Repeat until no improvement or max_iters.
    """
    aln = [list(s) for s in alignment]
    n = len(aln)
    L = len(aln[0])
    best_score = sp_score([''.join(row) for row in aln], match=match, mismatch=mismatch, gap=gap)
    improved = True
    it = 0
    while improved and it < max_iters:
        improved = False
        it += 1
        for seq_idx in range(n):
            for pos in range(L):
                if aln[seq_idx][pos] != '-':
                    continue
                # try move left
                if pos-1 >= 0:
                    # swap
                    aln[seq_idx][pos], aln[seq_idx][pos-1] = aln[seq_idx][pos-1], aln[seq_idx][pos]
                    cand_score = sp_score([''.join(row) for row in aln], match=match, mismatch=mismatch, gap=gap)
                    if cand_score > best_score:
                        best_score = cand_score
                        improved = True
                    else:
                        # revert
                        aln[seq_idx][pos], aln[seq_idx][pos-1] = aln[seq_idx][pos-1], aln[seq_idx][pos]
                # try move right
                if pos+1 < L:
                    aln[seq_idx][pos], aln[seq_idx][pos+1] = aln[seq_idx][pos+1], aln[seq_idx][pos]
                    cand_score = sp_score([''.join(row) for row in aln], match=match, mismatch=mismatch, gap=gap)
                    if cand_score > best_score:
                        best_score = cand_score
                        improved = True
                    else:
                        aln[seq_idx][pos], aln[seq_idx][pos+1] = aln[seq_idx][pos+1], aln[seq_idx][pos]
        if not improved:
            break
    return [''.join(row) for row in aln], best_score
