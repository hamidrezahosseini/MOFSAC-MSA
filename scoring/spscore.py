# scoring/spscore.py
def sp_score(alignment, match=2, mismatch=-1, gap=-2):
    """
    alignment: list of strings (aligned sequences, same length)
    Returns integer Sum-of-Pairs (SP) score with given scoring.
    """
    if not alignment:
        return 0
    n = len(alignment)
    L = len(alignment[0])
    # basic checks: all same length
    for s in alignment:
        if len(s) != L:
            raise ValueError("All aligned sequences must have same length")
    score = 0
    for col in range(L):
        for i in range(n):
            ai = alignment[i][col]
            for j in range(i+1, n):
                aj = alignment[j][col]
                if ai == '-' or aj == '-':
                    score += gap
                elif ai == aj:
                    score += match
                else:
                    score += mismatch
    return score
