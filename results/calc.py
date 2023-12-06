import numpy as np
from scipy.stats import ttest_ind

from plot import parse_file, file_paths

N_TOP = 10
Z_SCORE = 2.576
baseline_key = "VIG"
data = parse_file(file_paths)
top_n_test_baselines = sorted(data[baseline_key][1])[:N_TOP]


def trunc(x, n=5):
    return f"{x:0.100f}"[: n + 2]


for model in data:
    train, test = data[model]
    top_n_test_mse = sorted(test)[:N_TOP]

    # Get confidence interval
    mean = np.mean(top_n_test_mse)
    std_dev = np.std(top_n_test_mse, ddof=1)

    # Perform two-sample t-test
    t_statistic, p_value = ttest_ind(
        top_n_test_mse, top_n_test_baselines, alternative="less"
    )

    # Output the results
    print(f"Model: {model}")
    print(f"T-Statistic: {t_statistic}")
    print(f"P-Value: {p_value}")
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std_dev}")
    # print(f"CI: {mean:0.5f} $\pm$ {Z_SCORE * std_dev:0.5f} && {p_value:0.5f}")
    print(f"CI: {trunc(mean)} $\pm$ {trunc(Z_SCORE * std_dev)} & {trunc(p_value)}")
    print("--------------------------")
