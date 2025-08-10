import random

# set `noise` to False if you want to generate an answer
def dump(noise: bool = True) -> str:
    # len(seq) are all coprime, so that its cycle is very long
    seq1 = "abc"
    seq2 = "defg"
    seq3 = "hijkl"
    seq4 = "mnopqrs"
    seq5 = "tuvwxyz0123"
    chars = []

    for i in range(2000):
        chars.append(seq1[i % len(seq1)])
        chars.append(seq2[i % len(seq2)])
        chars.append(seq3[i % len(seq3)])

        # it appears twice hahaha
        chars.append(seq4[i % len(seq4)])
        chars.append(seq4[i % len(seq4)])

        chars.append(seq5[i % len(seq5)])
        chars.append(";")
        chars.append("\n")

    result = "".join(chars)

    # each character is replaced with a random character by 0.2%
    if noise:
        all_chars = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ;\n"
        result = "".join([c if random.random() > 0.002 else all_chars[random.randint(0, len(all_chars) - 1)] for c in result])

    return result

print(dump())
