

def get_max_acc(file):
    lines = []
    with open(file=file) as f:
        lines = f.readlines()
    ordered_lines = sorted(lines)
    print(ordered_lines[len(ordered_lines)-1])
    