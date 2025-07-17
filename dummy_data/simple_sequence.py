def dump():
    chars = "abcdefgh"
    delims = "@#$%^"
    result = []

    for i in range(2000):
        result.append(chars[i % len(chars)])
        result.append(delims[i % len(delims)])
        result.append(delims[i % len(delims)])

    return "".join(result)

print(dump(), end="")
