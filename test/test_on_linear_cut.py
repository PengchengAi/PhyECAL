import numpy as np
from scipy.stats import linregress, norm
import matplotlib.pyplot as plt


def main(result_path, thresh=0.7):
    result_content = np.load(result_path)
    out_result_list = result_content["result"]

    total_samples = out_result_list.shape[0]
    print("total samples: %d" % total_samples)

    x = np.arange(out_result_list.shape[1], dtype=np.float32)

    fit_result_list = []
    for i in range(total_samples):
        i_res = linregress(x, out_result_list[i, :])
        fit_result_list.append(i_res)

    r_value_list = [item.rvalue for item in fit_result_list]
    plt.figure()
    plt.title("Distribution of R value")
    plt.hist(r_value_list, bins=50)
    plt.axvline(x=thresh, color="r", linestyle="-.")
    plt.show()

    sel_result_list = [item for ind, item in enumerate(out_result_list) if r_value_list[ind] >= thresh]
    print("Selected samples: %d" % len(sel_result_list))

    cal_error_list = []
    for res in sel_result_list:
        for i in range(6):
            cal_error_list.append((res[i+2] + res[i] - res[i+1] * 2) * 0.4)

    std = np.std(cal_error_list)
    mean = np.mean(cal_error_list)
    std_equ = std / np.sqrt(6)

    plt.figure()
    plt.title("Distribution of differences\n(count: %d, equiv. std.: %.3f)" % (len(cal_error_list), std_equ))
    _, bins, _ = plt.hist(cal_error_list, bins=50, density=True)
    plt.plot(bins, norm.pdf(bins, loc=mean, scale=std), "r-.")
    plt.show()


if __name__ == "__main__":
    main(
        result_path="./temp/default_8ch/result/default_8ch-result.npz",
        thresh=-1.
    )
