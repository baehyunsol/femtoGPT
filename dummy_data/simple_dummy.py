from random import randint as ri

vowels = ["a", "e", "i", "o", "u"]
random_vowel = lambda: vowels[ri(0, len(vowels) - 1)]
consonants = ["b", "c", "d", "f", "g", "h", "j", "k"]
random_consonant = lambda: consonants[ri(0, len(consonants) - 1)]

def dump():
    result = ""
    last_3_prefixes = []

    for _ in range(2000):
        # - a word starts with a consonant
        # - recently used prefixes are not used again
        while (prefix := random_consonant()) in last_3_prefixes:
            pass

        curr_word = prefix
        last_3_prefixes.append(prefix)

        if len(last_3_prefixes) > 3:
            last_3_prefixes = last_3_prefixes[-3:]

        while True:
            # a consonant is follwed by a vowel and vice versa
            if curr_word[-1] in consonants:
                # a character is unique in a word
                while (v := random_vowel()) in curr_word:
                    pass

                curr_word += v

            else:
                while (c := random_consonant()) in curr_word:
                    pass

                curr_word += c

            if len(curr_word) == 4 and ri(0, 1) == 0 or len(curr_word) == 6:
                break

        result += curr_word
        result += ";\n"  # every word is followed by ";\n"

    return result

print(dump())
