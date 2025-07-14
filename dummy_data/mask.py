def mask(
    s: str,
    threshold: float = 0.001,
    replacement: str = "x",
) -> str:
    return "".join([c if random() > threshold else replacement for c in s])
