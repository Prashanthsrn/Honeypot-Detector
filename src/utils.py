def print_label_distribution(y, name="labels"):
    from collections import Counter
    counts = Counter(y)
    total = len(y)
    print(f"{name} distribution:")
    for k, v in counts.items():
        print(f"  {k}: {v} ({v/total:.2%})")
