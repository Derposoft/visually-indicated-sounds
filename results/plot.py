import re
import matplotlib.pyplot as plt


file_paths = {
    "VIG": "./output.train_vig_10bs",
    "D-VIG": "./output.train_dvig_1bs",
    "FoleyGAN": "./output.train_foleygan_1bs",
    "POCAN": "./output.train_pocan",
}


def parse_file(file_paths):
    data = {}

    for model in file_paths:
        file_path = file_paths[model]
        train_mse = []
        test_mse = []

        with open(file_path, "r") as file:
            lines = file.readlines()

        for line in lines:
            train_match = "Epoch" in line
            if train_match:
                train_mse.append(float(line.split(":")[-1]))

            test_match = "Test MSE:" in line
            if test_match:
                test_mse.append(float(line.split("[")[-1].split("]")[0]))

        data[model] = (train_mse, test_mse)

    return data


def plot_mse(data, label="Train MSE"):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for i, model in enumerate(data):
        train, test = data[model]
        d = test
        print(d[:5])
        axs[i % 2, i // 2].plot(d, label=model)
        axs[i % 2, i // 2].set_title(model)

        # plt.subplot(d, label=model)
    # plt.xlabel("Epoch")
    # plt.ylabel("MSE")
    # plt.title("")
    # plt.legend()
    plt.show()


if __name__ == "__main__":
    data = parse_file(file_paths)
    plot_mse(data, "Train MSE")
    # plot_mse(test_mse, "Test MSE")
