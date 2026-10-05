import re

pred = "Answer: B"
gold = " A"

# Many naive metrics do something like:
extracted = re.findall(r'[A-D]', pred)
print("Extracted from pred:", extracted)

# If target is A, and 'A' is in extracted:
target = gold.strip()
if target in extracted:
    print(f"Target {target} matched! Score: 1.0")
else:
    print(f"Target {target} not matched. Score: 0.0")
