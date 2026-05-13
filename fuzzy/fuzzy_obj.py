# fuzzy/fuzzy_obj.py
from scoring.spscore import sp_score

def compute_gap_count(alignment):
    return sum(row.count('-') for row in alignment)

def normalize(value, vmin, vmax):
    if vmax == vmin:
        return 0.0
    return max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))

def fuzzy_membership(alignment, match=2, mismatch=-1, gap=-2, weights=(0.6, 0.3, 0.1), approx_max_sp=None):
    """
    weights: tuple for SPscore, gap_count, (optional) computational-cost (ignored currently)
    approx_max_sp: approximate maximum possible SP for normalization (if None, estimate)
    returns aggregated membership in [0,1] (higher is better)
    """
    n = len(alignment)
    L = len(alignment[0]) if n>0 else 0
    sp = sp_score(alignment, match=match, mismatch=mismatch, gap=gap)
    # estimate maximum SP: if every pair is a match at every column -> max_sp = pairs * L * match
    pairs = n * (n-1) / 2
    max_sp = approx_max_sp if approx_max_sp is not None else pairs * L * match
    # normalize SP to [0,1]
    mu_sp = normalize(sp, vmin=-abs(max_sp), vmax=max_sp)  # allow negative lows
    # gap_count: fewer gaps better -> compute normalized inverse
    gapc = compute_gap_count(alignment)
    max_gaps = n * L  # worst case all gaps
    mu_gaps = 1.0 - normalize(gapc, 0, max_gaps)
    # cost term (placeholder)
    mu_cost = 1.0  # if we want to penalize long runtime, set accordingly
    w1, w2, w3 = weights
    total = w1 * mu_sp + w2 * mu_gaps + w3 * mu_cost
    # ensure in [0,1]
    return max(0.0, min(1.0, total)), {"mu_sp": mu_sp, "mu_gaps": mu_gaps}
