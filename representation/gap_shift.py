# representation/gap_shift.py
import copy
import math

def extract_gap_positions(alignment):
    """
    alignment: list of strings (seed alignment)
    returns:
      gap_info: list per sequence of gap positions (list of indices)
      total_gaps: total number of gaps across all sequences
    """
    gap_info = []
    for s in alignment:
        pos = [i for i, ch in enumerate(s) if ch == '-']
        gap_info.append(pos)
    total_gaps = sum(len(p) for p in gap_info)
    return gap_info, total_gaps

def decode_gap_shifts(seed_alignment, vector, max_shift=3):
    """
    seed_alignment: list of strings (same length)
    vector: list/array of floats length == total_gaps (ordering matches extract_gap_positions)
    max_shift: maximum allowed integer shift left/right for each gap (bounded)
    returns: new_alignment (list of strings) after applying deterministic shifts
    """
    # convert seed to list of lists for mutability
    aln = [list(s) for s in seed_alignment]
    gap_info, total = extract_gap_positions(seed_alignment)
    if len(vector) != total:
        raise ValueError("vector length mismatch with total gaps in seed")
    idx = 0
    # For each sequence, process its gaps left-to-right (indices in gap_info[seq])
    for seq_idx, gap_positions in enumerate(gap_info):
        for gap_pos in gap_positions:
            val = vector[idx]
            # map continuous val to integer shift in [-max_shift, max_shift]
            s = int(round(max(-max_shift, min(max_shift, val))))
            if s == 0:
                idx += 1
                continue
            # attempt to move this gap by s positions: positive -> move right, negative -> left
            # Implementation: swap gap char with neighbor chars step by step, ensuring we stay within bounds
            current_pos = gap_pos
            # But note: previous shifts may have changed positions; recompute current positions by finding next '-' occurrence near original
            # To avoid complexity, we will search for the nearest '-' starting at original index (leftmost occurrence not consumed)
            # Find the k-th '-' occurrence index at/after original gap_pos position
            # Simpler: search for a '-' in aln[seq_idx] whose original index equals gap_pos or nearest. We'll locate the first '-' at or after gap_pos.
            found_pos = None
            L = len(aln[seq_idx])
            for p in range(max(0, gap_pos-2*max_shift), min(L, gap_pos+2*max_shift+1)):
                if aln[seq_idx][p] == '-':
                    found_pos = p
                    break
            if found_pos is None:
                idx += 1
                continue
            current_pos = found_pos
            # perform stepwise shift
            steps = abs(s)
            direction = 1 if s > 0 else -1
            for step in range(steps):
                new_pos = current_pos + direction
                if new_pos < 0 or new_pos >= L:
                    break
                # swap positions in this sequence (move gap)
                aln[seq_idx][current_pos], aln[seq_idx][new_pos] = aln[seq_idx][new_pos], aln[seq_idx][current_pos]
                # when we move the gap in this sequence, other sequences must maintain column alignment: we just swap within that seq only,
                # but since column content changed we keep other sequences intact; lengths remain same.
                current_pos = new_pos
            idx += 1
    # convert back to strings
    new_alignment = [''.join(row) for row in aln]
    return new_alignment
