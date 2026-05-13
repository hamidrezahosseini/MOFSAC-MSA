# representation/hybrid_pro.py
import math
import numpy as np
from copy import deepcopy

def extract_gap_positions(alignment):
    gap_info = []
    for s in alignment:
        pos = [i for i, ch in enumerate(s) if ch == '-']
        gap_info.append(pos)
    total_gaps = sum(len(p) for p in gap_info)
    return gap_info, total_gaps

def build_hybrid_spec(seed_alignment, K_insert=8, K_delete=8, M_swaps=6, S_segments=6):
    """
    Determine dimension sizes for hybrid vector based on seed.
    Returns a dict with dims and helper metadata.
    """
    n = len(seed_alignment)
    L = len(seed_alignment[0])
    gap_info, total_gaps = extract_gap_positions(seed_alignment)
    spec = {
        "n": n, "L": L, "gap_info": gap_info, "total_gaps": total_gaps,
        "K_insert": K_insert, "K_delete": K_delete, "M_swaps": M_swaps, "S_segments": S_segments
    }
    # dims:
    dim = total_gaps + K_insert*2 + K_delete*2 + M_swaps*2 + S_segments*2
    # layout offsets
    offsets = {}
    off = 0
    offsets['gap_shifts'] = (off, off + total_gaps); off += total_gaps
    offsets['insert'] = (off, off + K_insert*2); off += K_insert*2
    offsets['delete'] = (off, off + K_delete*2); off += K_delete*2
    offsets['swaps'] = (off, off + M_swaps*2); off += M_swaps*2
    offsets['segments'] = (off, off + S_segments*2); off += S_segments*2
    spec['dim'] = dim
    spec['offsets'] = offsets
    return spec

def _to_int_in_range(x, lo, hi):
    # map continuous [-r,r] or [0,1] to integer in [lo,hi]
    if math.isnan(x):
        x = 0.0
    # first squash via sigmoid-like mapping
    # assume x in arbitrary range; use tanh to bound to [-1,1]
    v = math.tanh(float(x))
    # map [-1,1] -> [lo, hi]
    val = lo + int(round((v + 1.0) * 0.5 * (hi - lo)))
    return max(lo, min(hi, val))

def decode_hybrid(seed_alignment, vector, spec, max_shift=20):
    """
    Decode hybrid vector -> new alignment (list of strings).
    Steps:
    1) apply gap shifts (same as previous)
    2) apply gap insertions: each insert param -> (seq_idx, col_idx) where to insert a gap
    3) apply gap deletions: remove a gap at chosen (seq_idx, col_idx) if exists
    4) column swaps: swap entire columns i<->j
    5) segment ops: for each (start,end) do local realign by simple greedy (shift gaps inside segment)
    This decode is deterministic and robust (keeps lengths constant except insert/delete change).
    For insert/delete we will maintain column length by inserting a gap column (add column everywhere).
    """
    aln = [list(s) for s in deepcopy(seed_alignment)]
    n = spec['n']; L = spec['L']
    offsets = spec['offsets']
    total_gaps = spec['total_gaps']

    # 1) gap_shifts
    gs_lo, gs_hi = offsets['gap_shifts']
    # build flat list of gap positions order identical to extract_gap_positions
    gap_positions = []
    for seq_idx, poslist in enumerate(spec['gap_info']):
        for p in poslist:
            gap_positions.append((seq_idx, p))
    # apply shifts
    for k in range(gs_lo, gs_hi):
        val = vector[k]
        seq_idx, orig_pos = gap_positions[k - gs_lo]
        # find nearest '-' occurrence near original pos in current aln row
        row = aln[seq_idx]
        Lcur = len(row)
        found = None
        search_lo = max(0, orig_pos - max_shift)
        search_hi = min(Lcur-1, orig_pos + max_shift)
        # find k-th '-' occurrence based on order: look for nearest to orig_pos
        best = None
        best_dist = 10**9
        for p in range(search_lo, search_hi+1):
            if row[p] == '-':
                d = abs(p - orig_pos)
                if d < best_dist:
                    best_dist = d; best = p
        if best is None:
            continue
        shift = int(round(max(-max_shift, min(max_shift, float(val)))))
        cur_pos = best
        # stepwise swap
        dirc = 1 if shift > 0 else -1
        for s in range(abs(shift)):
            newp = cur_pos + dirc
            if newp < 0 or newp >= len(row): break
            # swap only in this row (columns move)
            row[cur_pos], row[newp] = row[newp], row[cur_pos]
            cur_pos = newp
        aln[seq_idx] = row

    # 2) gap insertions
    ins_lo, ins_hi = offsets['insert']
    K_insert = (ins_hi - ins_lo)//2
    for i in range(K_insert):
        a = vector[ins_lo + 2*i]   # controls seq index
        b = vector[ins_lo + 2*i+1] # controls column position
        seq_idx = _to_int_in_range(a, 0, n-1)
        col_idx = _to_int_in_range(b, 0, len(aln[0]))  # insertion can be at end
        # insert global column at col_idx filled with '-' then set that seq to original char shift
        for sidx in range(n):
            aln[sidx].insert(col_idx, '-')
        # leave inserted gap at seq_idx (a gap inserted in other sequences), or we could keep char
        # (better: keep as gap in all but maintain some char shift) — here keep gap in all.
        # update lengths consistent

    # 3) gap deletions (attempt to remove an existing '-' in that seq at position)
    del_lo, del_hi = offsets['delete']
    K_delete = (del_hi - del_lo)//2
    for i in range(K_delete):
        a = vector[del_lo + 2*i]
        b = vector[del_lo + 2*i+1]
        seq_idx = _to_int_in_range(a, 0, n-1)
        col_idx = _to_int_in_range(b, 0, len(aln[0])-1)
        if aln[seq_idx][col_idx] == '-':
            # remove this column only if majority are gaps? else we convert this position to char by shifting neighbor
            # simple strategy: delete the column across all sequences (removes a gap column)
            for sidx in range(n):
                aln[sidx].pop(col_idx)

    # 4) column swaps
    sw_lo, sw_hi = offsets['swaps']
    M_swaps = (sw_hi - sw_lo)//2
    for i in range(M_swaps):
        a = vector[sw_lo + 2*i]; b = vector[sw_lo + 2*i+1]
        col1 = _to_int_in_range(a, 0, len(aln[0])-1)
        col2 = _to_int_in_range(b, 0, len(aln[0])-1)
        if col1 != col2:
            for sidx in range(n):
                aln[sidx][col1], aln[sidx][col2] = aln[sidx][col2], aln[sidx][col1]

    # 5) segment ops - basic local shuffle within segment
    seg_lo, seg_hi = offsets['segments']
    S_segments = (seg_hi - seg_lo)//2
    for i in range(S_segments):
        a = vector[seg_lo + 2*i]; b = vector[seg_lo + 2*i+1]
        start = _to_int_in_range(a, 0, max(0, len(aln[0])-2))
        end = _to_int_in_range(b, start+1, len(aln[0])-1)
        # simple operation: reverse columns in [start,end]
        for sidx in range(n):
            seg = aln[sidx][start:end+1]
            seg.reverse()
            aln[sidx][start:end+1] = seg

    # convert back to strings
    new_alignment = [''.join(row) for row in aln]
    return new_alignment
