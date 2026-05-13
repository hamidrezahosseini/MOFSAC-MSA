# seed_generation.py
def generate_seed_alignment(sequences):
    """
    Seed alignment ساده (Baseline):
    سکانس‌ها را بدون تغییر محتوا، فقط هم‌طول می‌کند
    """

    if not sequences:
        return []

    max_len = max(len(seq) for seq in sequences)

    seed_alignment = []
    for seq in sequences:
        # Padding با گپ در انتها
        padded = seq.ljust(max_len, '-')
        seed_alignment.append(padded)

    return seed_alignment
