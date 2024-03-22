import json
import os
import sys

if len(sys.argv) != 2:
    print("Usage: python parse_results.py <experiment_name>")
    exit(1)

results_dir = "/cs/student/projects3/COMP0087/grp2/clean/experiments/" + sys.argv[1]

total_samples = 0
good_samples = 0

for filename in os.listdir(results_dir):
    if filename.endswith("results.json"):
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)

            for sample_id, sample_data in data.items():
                total_samples += 1
                if sample_data["Status"] == "Good":
                    good_samples += 1

percentage_good = (good_samples / total_samples) * 100

# Store the result (adjust as needed)
print(f"Percentage of Good samples: {percentage_good:.2f}%") 