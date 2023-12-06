import re
import matplotlib.pyplot as plt


def parse_file(file_path):
    train_mse = []
    test_mse = []

    with open(file_path, "r") as file:
        lines = file.readlines()
    print(lines[0])

    for line in lines:
        train_match = "Epoch" in line
        if train_match:
            train_mse.append(float(line.split(":")[-1]))

        test_match = "Test MSE:" in line
        if test_match:
            test_mse.append(float(line.split("[")[1].split("]")[0]))

    return train_mse, test_mse


def plot_mse(mse, label="Train MSE"):
    plt.plot(mse, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("")
    # plt.legend()
    plt.show()


if __name__ == "__main__":
    file_path = "./output.train_dvig_1bs"
    train_mse, test_mse = parse_file(file_path)
    plot_mse(train_mse, "Train MSE")
    plot_mse(test_mse, "Test MSE")
