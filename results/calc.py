import numpy as np
from scipy.stats import ttest_ind
import glob

N_TOP = 10
Z_SCORE = 1.68
VIG_BASELINES = [
    0.13182008266448975,
    0.13182008266448975,
    0.13182008266448975,
    0.13182008266448975,
    0.13182008266448975,
    0.13182008266448975,
    0.13182008266448975,
    0.13182008266448975,
    0.13182008266448975,
    0.13182009756565094,
]

# Specify the pattern to match files starting with "output"
files_starting_with_output = glob.glob("output*")

# Output the list of matching files
print(files_starting_with_output)

for file_name in files_starting_with_output:
    if file_name.endswith("vig"):
        continue
    print("file_name:", file_name)
    with open(file_name, "r") as file:
        data = file.read()

    # Split the data by lines
    lines = data.split("\n")

    # Extract Test MSE values
    test_mse_values = []
    for line in lines:
        if "Test MSE" in line:
            mse = float(line.split("[", 1)[1].split("]")[0])
            test_mse_values.append(mse)

    # Sort the Test MSE values in ascending order
    test_mse_values.sort()

    top_n_test_mse = test_mse_values[:N_TOP]
    # print(top_n_test_mse)

    # Calculate mean and standard deviation
    mean = np.mean(top_n_test_mse)
    std_dev = np.std(top_n_test_mse, ddof=1)

    # Perform two-sample t-test
    t_statistic, p_value = ttest_ind(VIG_BASELINES, top_n_test_mse, alternative="less")

    # Output the results
    print(f"T-Statistic: {t_statistic}")
    print(f"P-Value: {p_value}")

    # Output the results
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std_dev}")
    print(f"CI: {mean} $\pm$ {Z_SCORE * std_dev}")

    print("--------------------------")
