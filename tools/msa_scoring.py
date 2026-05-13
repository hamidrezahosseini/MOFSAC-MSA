# tools/msa_scoring.py

def sp_score(alignment, match=2, mismatch=-1, gap=-2):
    """
    Computes SP (Sum-of-Pairs) score for a multiple sequence alignment.

    Parameters
    ----------
    alignment : list of str
        List of aligned sequences (all must have equal length).
    match : int, optional
        Score for a character match (default: 2).
    mismatch : int, optional
        Score for a character mismatch (default: -1).
    gap : int, optional
        Penalty for a gap (default: -2).

    Returns
    -------
    int
        Sum-of-pairs score over all columns.
    """

    n = len(alignment)
    L = len(alignment[0])
    score = 0

    for pos in range(L):
        col = [alignment[i][pos] for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                a = col[i]
                b = col[j]

                if a == '-' or b == '-':
                    score += gap
                else:
                    score += match if a == b else mismatch

    return score


def cs_score(alignment):
    """
    Computes Column Score (CS) = EM / AL.

    EM: Number of exact-match columns where all non-gap
        characters in the column are identical.
    AL: Alignment length (total number of columns).

    Fully-gap columns are ignored for EM counting but
    still contribute to AL, following standard practice.

    Parameters
    ----------
    alignment : list of str
        List of aligned sequences.

    Returns
    -------
    float
        Column Score in [0, 1].
    """

    n = len(alignment)
    L = len(alignment[0])

    if L == 0:
        return 0.0

    exact_matches = 0

    for pos in range(L):
        col = [alignment[i][pos] for i in range(n)]
        non_gaps = [c for c in col if c != '-']

        # ignore fully-gap columns for EM
        if not non_gaps:
            continue

        if len(set(non_gaps)) == 1:
            exact_matches += 1

    return exact_matches / L


def gap_ratio(alignment):
    """
    Computes the gap percentage (Gap %) of a multiple sequence alignment.

    Gap % is defined as the ratio of gap characters ('-') to
    the total number of alignment cells.

        Gap (%) = (# of '-' characters) /
                  (num_sequences × alignment_length) × 100

    This metric is reference-free and is commonly used to
    quantify gap inflation in MSA optimization.

    Parameters
    ----------
    alignment : list of str
        List of aligned sequences.

    Returns
    -------
    float
        Gap percentage in [0, 100].
    """

    if not alignment:
        return 0.0

    num_seqs = len(alignment)
    aln_len = len(alignment[0])

    if aln_len == 0:
        return 0.0

    total_cells = num_seqs * aln_len
    gap_count = sum(seq.count('-') for seq in alignment)

    return 100.0 * gap_count / total_cells


def alignment_length(alignment):
    """
    Returns the alignment length (number of columns).

    Parameters
    ----------
    alignment : list of str

    Returns
    -------
    int
        Alignment length.
    """

    if not alignment:
        return 0

    return len(alignment[0])
