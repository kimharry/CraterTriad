# check triad db length

with open('data/crater_index_db1.csv', 'r') as f:
    lines = f.readlines()
    print(f"Number of lines: {len(lines) - 1}")

with open('data/crater_index_db2.csv', 'r') as f:
    lines = f.readlines()
    print(f"Number of lines: {len(lines) - 1}")

with open('data/crater_index_db3.csv', 'r') as f:
    lines = f.readlines()
    print(f"Number of lines: {len(lines) - 1}")