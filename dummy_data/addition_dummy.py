import random

def dump():
    result = []

    for _ in range(2000):
        a, b = random.randint(0, 100), random.randint(0, 100)
        result.append(f"{a} + {b} = {a + b};")

    return "\n".join(result)

print(dump())
