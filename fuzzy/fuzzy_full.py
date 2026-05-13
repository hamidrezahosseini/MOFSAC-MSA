# fuzzy/fuzzy_full.py
from scoring.spscore import sp_score
import math

def pairwise_counts(alignment):
    """
    alignment: list of equal-length strings
    returns:
      n: number of sequences
      L: alignment length (columns)
      pairs: number of pairs n*(n-1)/2
      mismatch_count: number of pairwise mismatches (excluding gaps)
      gap_pair_penalty: sum over pairwise gap penalties (positive number)
      total_gaps: total count of '-' characters across all sequences
      aln_score: Sum-of-Pairs alignment score using (match=+2, mismatch=-1, gap=-2)
    """
    n = len(alignment)
    if n == 0:
        return 0,0,0,0,0,0,0
    L = len(alignment[0])
    for s in alignment:
        if len(s) != L:
            raise ValueError("All sequences in alignment must have same length")
    pairs = n * (n-1) // 2

    mismatch_count = 0
    gap_pair_penalty = 0  # positive aggregate penalty (abs value)
    total_gaps = 0

    match = 2
    mismatch = -1
    gap = -2

    for col in range(L):
        col_chars = [alignment[i][col] for i in range(n)]
        # count total gaps in this column
        total_gaps += col_chars.count('-')
        # pairwise
        for i in range(n):
            ai = col_chars[i]
            for j in range(i+1, n):
                aj = col_chars[j]
                if ai == '-' or aj == '-':
                    gap_pair_penalty += abs(gap)  # treat penalty as positive
                elif ai != aj:
                    mismatch_count += 1
                else:
                    pass  # match

    # compute alignment score using sp_score helper (already sums signed contributions)
    aln_score = sp_score(alignment, match=match, mismatch=mismatch, gap=gap)

    return n, L, pairs, mismatch_count, gap_pair_penalty, total_gaps, aln_score

# membership helper functions:
def decreasing_linear(x, a, b):
    """ μ(x) = 1 if x <= a; linear down to 0 at x=b; 0 if x>=b """
    if b <= a:
        return 0.0
    if x <= a:
        return 1.0
    if x >= b:
        return 0.0
    return (b - x) / (b - a)

def increasing_linear(x, a, b):
    """ μ(x) = 0 if x <= a; linear up to 1 at x=b; 1 if x>=b """
    if b <= a:
        return 0.0
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a)

def triangular(x, a, b, c):
    """
    triangular membership:
    0 if x <= a
    linear up to 1 at b (a < x < b)
    linear down to 0 at c (b < x < c)
    0 if x >= c
    """
    if not (a < b < c):
        # invalid params: return 0 clipped
        if x == b:
            return 1.0
        return 0.0
    if x <= a or x >= c:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a)
    if b <= x < c:
        return (c - x) / (c - b)
    return 0.0

def fuzzy_total_membership(alignment,
                           weights=(0.3, 0.3, 0.3, 0.1),
                           match_val=2, mismatch_val=-1, gap_val=-2,
                           # thresholds will be computed from alignment size; the defaults below are multipliers
                           b1_frac=0.20,    # b1 = b1_frac * max_mismatches
                           b2_frac=0.20,    # b2 = b2_frac * max_gap_penalty
                           a3_frac=0.50,    # a3 = a3_frac * max_possible_score
                           b3_frac=0.80,    # b3 = b3_frac * max_possible_score
                           a4_frac=0.10,    # a4 = a4_frac * n*L (total gaps)
                           b4_frac=0.30,    # b4 = b4_frac * n*L
                           c_frac=0.60      # c = c_frac * n*L
                           ):
    """
    Compute μf1..μf4 and aggregated μftotal.
    thresholds are derived from alignment size and the fractional parameters above.
    Returns: (mu_total, details_dict)
    details_dict includes mu1..mu4, raw metrics and thresholds used.
    """
    n, L, pairs, mismatch_count, gap_pair_penalty, total_gaps, aln_score = pairwise_counts(alignment)

    # edge cases
    if n == 0 or L == 0 or pairs == 0:
        return 0.0, {"reason": "empty alignment"}

    # maximum possibles
    max_mismatches = pairs * L            # if every pair in every column is mismatch
    max_gap_penalty = pairs * L * abs(gap_val)  # if every pair in every column involves a gap
    max_possible_score = pairs * L * match_val  # if every pair matches in every column

    # thresholds
    a1 = 0.0
    b1 = max(1.0, b1_frac * max_mismatches)   # avoid zero
    a2 = 0.0
    b2 = max(1.0, b2_frac * max_gap_penalty)
    a3 = a3_frac * max_possible_score
    b3 = max(a3 + 1e-6, b3_frac * max_possible_score)   # ensure b3 > a3
    a4 = a4_frac * (n * L)
    b4 = max(a4 + 1e-6, b4_frac * (n * L))
    c = max(b4 + 1e-6, c_frac * (n * L))

    # compute memberships
    mu1 = decreasing_linear(mismatch_count, a1, b1)         # low mismatches -> high membership
    mu2 = decreasing_linear(gap_pair_penalty, a2, b2)      # low gap penalty -> high membership
    mu3 = increasing_linear(aln_score, a3, b3)             # high alignment score -> high membership
    mu4 = triangular(total_gaps, a4, b4, c)                # cost prefer around b4

    w1, w2, w3, w4 = weights
    # normalize weights sum to 1 (just in case user passes slightly non-normalized)
    wsum = w1 + w2 + w3 + w4
    if wsum == 0:
        w1 = 0.25; w2 = 0.25; w3 = 0.25; w4 = 0.25
    else:
        w1 /= wsum; w2 /= wsum; w3 /= wsum; w4 /= wsum

    mu_total = w1 * mu1 + w2 * mu2 + w3 * mu3 + w4 * mu4

    details = {
        "n": n, "L": L, "pairs": pairs,
        "mismatch_count": mismatch_count,
        "gap_pair_penalty": gap_pair_penalty,
        "total_gaps": total_gaps,
        "alignment_score": aln_score,
        "thresholds": {
            "a1": a1, "b1": b1,
            "a2": a2, "b2": b2,
            "a3": a3, "b3": b3,
            "a4": a4, "b4": b4, "c": c
        },
        "memberships": {"mu1": mu1, "mu2": mu2, "mu3": mu3, "mu4": mu4},
        "weights": {"w1": w1, "w2": w2, "w3": w3, "w4": w4},
        "mu_total": mu_total
    }
    return mu_total, details

# quick test runner if executed directly
if __name__ == "__main__":
    # small sanity test
    aln = [
        "A-UACG-",
        "AUUA-GG",
        "A-UACG-"
    ]
    mu, d = fuzzy_total_membership(aln)
    print("mu_total:", mu)
    print("details:", d)
