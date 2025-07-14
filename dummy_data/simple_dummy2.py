from random import randint as ri

# A line consists of 3 parts: data, action and result. A line must ends with ";".
# - Data: random generated string: `[a-f]{4, 8}`
# - Action: a decimal number
#   - 0: immediately terminates the line
#   - 1: copies the first 3 characters of the data
#   - 2: copies the last 3 characters of the data
#   - 3: copies the first 3 characters of the data, and maps each character like below
#     - "a" -> "g", "b" -> "h", "c" -> "i", "d" -> "j", "e" -> "k", "f" -> "l"
#   - 4: copies the last 3 characters of the data, and maps each character like below
#     - "a" -> "g", "b" -> "h", "c" -> "i", "d" -> "j", "e" -> "k", "f" -> "l"
#   - 5: copies the first 3 characters of the data of the previous line
#   - 6: copies the last 3 characters of the data of the previous line
#   - 7: copies the first 3 characters of the data of the previous line, and maps each character like below
#     - "a" -> "g", "b" -> "h", "c" -> "i", "d" -> "j", "e" -> "k", "f" -> "l"
#   - 8: copies the last 3 characters of the data of the previous line, and maps each character like below
#     - "a" -> "g", "b" -> "h", "c" -> "i", "d" -> "j", "e" -> "k", "f" -> "l"
#   - 9: dumps "z"
# - Result: result of the action
def dump():
    lines = []
    previous_data = None

    for _ in range(50000):
        data = "".join([chr(ord("a") + ri(0, 5)) for _ in range(ri(4, 8))])
        action = ri(0, 9)

        if previous_data is None and action in [5, 6, 7, 8]:
            continue

        if action == 0:
            result = ""
        elif action == 1:
            result = data[:3]
        elif action == 2:
            result = data[-3:]
        elif action == 3:
            result = "".join([chr(ord(c) + 6) for c in data[:3]])
        elif action == 4:
            result = "".join([chr(ord(c) + 6) for c in data[-3:]])
        elif action == 5:
            result = previous_data[:3]
        elif action == 6:
            result = previous_data[-3:]
        elif action == 7:
            result = "".join([chr(ord(c) + 6) for c in previous_data[:3]])
        elif action == 8:
            result = "".join([chr(ord(c) + 6) for c in previous_data[-3:]])
        elif action == 9:
            result = "z"

        lines.append(data + str(action) + result + ";")
        previous_data = data

    return "\n".join(lines)

def eval():
    import re
    import subprocess

    eval_data = dump()
    samples = []
    previous_lines = []

    for line in eval_data.split("\n"):
        r = re.match(r"[a-f]{4,8}(\d)([a-z]*\;)", line)
        action = int(r.group(1))
        answer = r.group(2) + "\n"

        if len(previous_lines) > 3:
            samples.append((
                "\n".join(previous_lines + [line[:-len(answer) + 1]]),
                action,
                line + "\n",
            ))

        previous_lines.append(line)

        if len(previous_lines) > 10:
            previous_lines = previous_lines[-10:]

        if len(samples) == 100:
            break

    correct = {}
    incorrect = {}

    for prompt, action, answer in samples:
        prompt = prompt[-60:]
        print("--- @@@@@@ ---")
        print("--- prompt ---")
        print(prompt)
        print("--- answer ---")
        print(answer)
        response = subprocess.run(["cargo", "run", "--release", "--", "infer", prompt], text=True, capture_output=True).stdout
        print("--- response ---")
        print(response)

        if answer in response:
            print("correct!")
            correct[action] = correct.get(action, 0) + 1

        else:
            print("incorrect!")
            incorrect[action] = incorrect.get(action, 0) + 1

    print("correct:", correct, "incorrect:", incorrect)

# eval()
print(dump())
