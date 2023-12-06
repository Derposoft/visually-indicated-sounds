import numpy as np
from scipy.stats import ttest_ind
import glob

# Specify the pattern to match files starting with "output"
files_starting_with_output = glob.glob('output*')

# Output the list of matching files
print(files_starting_with_output)

for file_name in files_starting_with_output:
    print("file_name:", file_name)
    with open(file_name, 'r') as file:
        data = file.read()

    # Split the data by lines
    lines = data.split('\n')

    # Extract Test MSE values
    test_mse_values = []
    for line in lines:
        if 'Test MSE' in line:
            mse = float(line.split('[', 1)[1].split(']')[0])
            test_mse_values.append(mse)

    # Sort the Test MSE values in ascending order
    test_mse_values.sort()

    current_top_5_test_mse = test_mse_values[:5]
    print(current_top_5_test_mse)

    # Calculate mean and standard deviation
    mean = np.mean(current_top_5_test_mse)
    std_dev = np.std(current_top_5_test_mse, ddof=1)

    top_5_test_mse_set1 = [1.3199845552444458, 1.319984793663025, 1.319984793663025, 1.319984793663025, 1.319985032081604] #change

    # Perform two-sample t-test
    t_statistic, p_value = ttest_ind(top_5_test_mse_set1, current_top_5_test_mse)

    # Output the results
    print(f"T-Statistic: {t_statistic}")
    print(f"P-Value: {p_value}")

    # Output the results
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std_dev}")

    print("--------------------------")
