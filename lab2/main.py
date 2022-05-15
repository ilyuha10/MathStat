import math
import os

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import scipy.stats as stats
from matplotlib.patches import Ellipse
from scipy.optimize import minimize
from scipy.stats import chi2, t, norm, moment


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor,
                      edgecolor='midnightblue', **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def task5():
    sizes = [20, 60, 100]
    rhos = [0, 0.5, 0.9]

    if not os.path.exists("task5_data"):
        os.mkdir("task5_data")

    # Count params for table
    # count for 2d normal
    for size in sizes:
        rows = []
        for rho in rhos:
            rows.append(['rho = ' + str(rho), 'r ',
                         'r_Q ', 'r_S '])
            Pirson = []
            Spirman = []
            SquaredCorrelation = []

            for repeat in range(0, 1000):
                sample_d = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=size)

                cov = np.cov(sample_d[:, 0], sample_d[:, 1])
                Pirson.append(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))

                n1, n2, n3, n4 = 0, 0, 0, 0
                for el in sample_d:
                    if el[0] >= 0 and el[1] >= 0:
                        n1 += 1
                    elif el[0] < 0 and el[1] >= 0:
                        n2 += 1
                    elif el[0] < 0 and el[1] < 0:
                        n3 += 1
                    else:
                        n4 += 1

                Spirman.append(stats.spearmanr(sample_d[:, 0], sample_d[:, 1])[0])
                SquaredCorrelation.append((n1 - n2 + n3 - n4) / len(sample_d))
            rows.append(['E(z)', np.mean(Pirson), np.mean(Spirman), np.mean(SquaredCorrelation)])
            rows.append(['E(z^2)', np.mean(np.asarray([el * el for el in Pirson])),
                         np.mean(np.asarray([el * el for el in Spirman])),
                         np.mean(np.asarray([el * el for el in SquaredCorrelation]))])
            rows.append(['D(z)', np.std(Pirson), np.std(Spirman), np.std(SquaredCorrelation)])
            if rho != rhos[-1]:
                rows.append(['', '', '', ''])
        with open("task5_data/task5_" + str(size) + ".txt", "w") as f:
            for row in rows:
                f.write(" ".join([str(i) for i in row]) + "\n")

    # count for mix normal
    for size in sizes:
        rows = []
        rows.append(['n = ' + str(size), 'r ',
                     'r_Q', 'r_S '])
        Pirson = []
        Spirman = []
        SquaredCorrelation = []

        for repeat in range(0, 1000):
            sample_d = 0.9 * np.random.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], size=size) + \
                       0.1 * np.random.multivariate_normal([0, 0], [[10, 0.9], [0.9, 10]], size=size)

            cov = np.cov(sample_d[:, 0], sample_d[:, 1])
            Pirson.append(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))

            n1, n2, n3, n4 = 0, 0, 0, 0
            for el in sample_d:
                if el[0] >= 0 and el[1] >= 0:
                    n1 += 1
                elif el[0] < 0 and el[1] >= 0:
                    n2 += 1
                elif el[0] < 0 and el[1] < 0:
                    n3 += 1
                else:
                    n4 += 1

            Spirman.append(stats.spearmanr(sample_d[:, 0], sample_d[:, 1])[0])
            SquaredCorrelation.append((n1 - n2 + n3 - n4) / len(sample_d))
        rows.append(['$E(z)$', np.mean(Pirson), np.mean(Spirman), np.mean(SquaredCorrelation)])
        rows.append(['$E(z^2)$', np.mean(np.asarray([el * el for el in Pirson])),
                     np.mean(np.asarray([el * el for el in Spirman])),
                     np.mean(np.asarray([el * el for el in SquaredCorrelation]))])
        rows.append(['$D(z)$', np.std(Pirson), np.std(Spirman), np.std(SquaredCorrelation)])
        if rho != rhos[-1]:
            rows.append(['', '', '', ''])
    with open("task5_data/task5_mix.txt", "w") as f:
        for row in rows:
            f.write(" ".join([str(i) for i in row]) + "\n")

    # Draw ellipses
    for size in sizes:
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        for ax, rho in zip(axs, rhos):
            sample_d = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=size)
            ax.scatter(sample_d[:, 0], sample_d[:, 1], color='red', s=3)
            ax.set_xlim(min(sample_d[:, 0]) - 2, max(sample_d[:, 0]) + 2)
            ax.set_ylim(min(sample_d[:, 1]) - 2, max(sample_d[:, 1]) + 2)
            print(min(sample_d[:, 0]), max(sample_d[:, 0]))
            print(min(sample_d[:, 1]), max(sample_d[:, 1]))
            ax.axvline(c='grey', lw=1)
            ax.axhline(c='grey', lw=1)
            title = r'n = ' + str(size) + r', rho  = ' + str(rho)
            ax.set_title(title)
            confidence_ellipse(sample_d[:, 0], sample_d[:, 1], ax)
        fig.savefig('task5_data/' + str(size) + '.png', dpi=200)

    # Draw mixed ellipse
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    for ax, size in zip(axs, sizes):
        sample_d = 0.9 * np.random.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], size=size) + \
                   0.1 * np.random.multivariate_normal([0, 0], [[10, -0.9], [-0.9, 10]], size=size)
        ax.scatter(sample_d[:, 0], sample_d[:, 1], color='red', s=3)
        ax.set_xlim(min(sample_d[:, 0]) - 2, max(sample_d[:, 0]) + 2)
        ax.set_ylim(min(sample_d[:, 1]) - 2, max(sample_d[:, 1]) + 2)
        print(min(sample_d[:, 0]), max(sample_d[:, 0]))
        print(min(sample_d[:, 1]), max(sample_d[:, 1]))
        ax.axvline(c='grey', lw=1)
        ax.axhline(c='grey', lw=1)
        title = r'mixed: n = ' + str(size)
        ax.set_title(title)
        confidence_ellipse(sample_d[:, 0], sample_d[:, 1], ax)
    fig.savefig('task5_data/mix.png', dpi=200)
    plt.show()


def get_least_squares(x, y):
    xy_med = np.mean(np.multiply(x, y))
    x_med = np.mean(x)
    x_2_med = np.mean(np.multiply(x, x))
    y_med = np.mean(y)
    b1_mnk = (xy_med - x_med * y_med) / (x_2_med - x_med * x_med)
    b0_mnk = y_med - x_med * b1_mnk

    dev = 0
    for i in range(len(x)):
        dev += (b0_mnk + b1_mnk * x[i] - y[i]) ** 2
    print('Невязка МНК:' + str(math.sqrt(dev)))
    return b0_mnk, b1_mnk


def abs_dev_val(b_arr, x, y):
    return np.sum(np.abs(y - b_arr[0] - b_arr[1] * x))


def get_linear_approx(x, y):
    init_b = np.array([0, 1])
    res = minimize(abs_dev_val, init_b, args=(x, y), method='COBYLA')

    dev = 0
    for i in range(len(x)):
        dev += math.fabs(res.x[0] + res.x[1] * x[i] - y[i])
    return res.x


def draw(lsm_0, lsm_1, lam_0, lam_1, x, y, title, fname):
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', s=6, label='Выборка')
    y_lsm = np.add(np.full(20, lsm_0), x * lsm_1)
    y_lam = np.add(np.full(20, lam_0), x * lam_1)
    y_real = np.add(np.full(20, 2), x * 2)
    ax.plot(x, y_lsm, color='blue', label='МНК')
    ax.plot(x, y_lam, color='red', label='МНМ')
    ax.plot(x, y_real, color='green', label='Модель')
    ax.set(xlabel='X', ylabel='Y',
           title=title)
    ax.legend()
    ax.grid()
    fig.savefig(fname + '.png', dpi=200)


def task6():
    # values
    x = np.arange(-1.8, 2.1, 0.2)
    # error
    eps = np.random.normal(0, 1, size=20)

    # y = 2 + 2x + eps
    y = np.add(np.add(np.full(20, 2), x * 2), eps)
    # y2 same but with noise
    y2 = np.add(np.add(np.full(20, 2), x * 2), eps)
    y2[0] += 10
    y2[19] += -10

    lsm_0, lsm_1 = get_least_squares(x, y)
    print("МНК, без возмущений: " + str(lsm_0) + ", " + str(lsm_1))
    lam_0, lam_1 = get_linear_approx(x, y)
    print("МНМ, без возмущений: " + str(lam_0) + ", " + str(lam_1))

    lsm_02, lsm_12 = get_least_squares(x, y2)
    print("МНК, с возмущенями: " + str(lsm_02) + ", " + str(lsm_12))
    lam_02, lam_12 = get_linear_approx(x, y2)
    print("МНМ, с возмущенями: " + str(lam_02) + ", " + str(lam_12))

    if not os.path.exists("task6_data"):
        os.mkdir("task6_data")

    draw(lsm_0, lsm_1, lam_0, lam_1, x, y, 'Выборка без возмущений', 'task6_data/no_dev')
    draw(lsm_02, lsm_12, lam_02, lam_12, x, y2, 'Выборка с возмущенями', 'task6_data/dev')
    plt.show()


def get_cdf(x, mu, sigma):
    return stats.norm.cdf(x, mu, sigma)


def normal_exp():
    sz = 100
    samples = np.random.normal(0, 1, size=sz)
    mu_c = np.mean(samples)
    sigma_c = np.std(samples)
    alpha = 0.05
    p = 1 - alpha
    k = 7
    left = mu_c - 3
    right = mu_c + 3
    borders = np.linspace(left, right, num=(k - 1))
    value = chi2.ppf(p, k - 1)
    p_arr = np.array([get_cdf(borders[0], mu_c, sigma_c)])
    for i in range(len(borders) - 1):
        val = get_cdf(borders[i + 1], mu_c, sigma_c) - get_cdf(borders[i], mu_c, sigma_c)
        p_arr = np.append(p_arr, val)
    p_arr = np.append(p_arr, 1 - get_cdf(borders[-1], mu_c, sigma_c))
    print(f"mu: " + str(mu_c) + ", sigma: " + str(sigma_c))
    print(f"Промежутки: " + str(borders) + "\n"
                                           f"p_i: \n " + str(p_arr) + " \n"
                                                                      f"n * p_i: \n " + str(p_arr * sz) + "\n"
                                                                                                          f"Сумма: " + str(
        np.sum(p_arr)) + "\n")
    n_arr = np.array([len(samples[samples <= borders[0]])])
    for i in range(len(borders) - 1):
        n_arr = np.append(n_arr, len(samples[(samples <= borders[i + 1]) & (samples >= borders[i])]))
    n_arr = np.append(n_arr, len(samples[samples >= borders[-1]]))
    res_arr = np.divide(np.multiply((n_arr - p_arr * 100), (n_arr - p_arr * 100)), p_arr * 100)
    print(f"n_i: \n" + str(n_arr) + "\n"
                                    f"Сумма: " + str(np.sum(n_arr)) + "\n"
                                                                      f"n_i  - n*p_i: " + str(
        n_arr - p_arr * sz) + "\n"
                              f"res: " + str(res_arr) + "\n"
                                                        f"res_sum = " + str(np.sum(res_arr)) + "\n")

    rows = [[i + 1, ("%.2f" % borders[(i + 1) % (k - 1)]) + ', ' + "%.2f" % borders[(i + 2) % (k - 1)],
             "%.2f" % n_arr[i],
             "%.4f" % p_arr[i],
             "%.2f" % (sz * p_arr[i]),
             "%.2f" % (n_arr[i] - sz * p_arr[i]),
             "%.2f" % ((n_arr[i] - sz * p_arr[i]) ** 2 / (sz * p_arr[i]))] for i in range(k)]
    rows.insert(0, ['line i', 'Границы Delta_i', 'n_i', 'p_i', 'np_i' 'n_i - np_i',
                    '(n_i - np_i)^2/np_i'])
    with open("task7_data/task7_normal.txt", "w") as f:
        for row in rows:
            f.write(" ".join([str(i) for i in row]) + "\n")


def laplace_exp():
    sz = 20
    samples = np.random.laplace(0, 1 / math.sqrt(2), size=sz)
    mu_c = np.mean(samples)
    sigma_c = np.std(samples)

    alpha = 0.05
    p = 1 - alpha
    k = 7
    left = mu_c - 3
    right = mu_c + 3
    borders = np.linspace(left, right, num=(k - 1))
    p_arr = np.array([get_cdf(borders[0], mu_c, sigma_c)])
    for i in range(len(borders) - 1):
        val = get_cdf(borders[i + 1], mu_c, sigma_c) - get_cdf(borders[i], mu_c, sigma_c)
        p_arr = np.append(p_arr, val)
    p_arr = np.append(p_arr, 1 - get_cdf(borders[-1], mu_c, sigma_c))
    print(f"mu: " + str(mu_c) + ", sigma: " + str(sigma_c))
    print(f"Промежутки: " + str(borders) + "\n"
                                           f"p_i: \n " + str(p_arr) + " \n"
                                                                      f"n * p_i: \n " + str(p_arr * sz) + "\n"
                                                                                                          f"Сумма: " + str(
        np.sum(p_arr)) + "\n")
    n_arr = np.array([len(samples[samples <= borders[0]])])
    for i in range(len(borders) - 1):
        n_arr = np.append(n_arr, len(samples[(samples <= borders[i + 1]) & (samples >= borders[i])]))
    n_arr = np.append(n_arr, len(samples[samples >= borders[-1]]))
    res_arr = np.divide(np.multiply((n_arr - p_arr * sz), (n_arr - p_arr * sz)), p_arr * sz)
    print(f"n_i: \n" + str(n_arr) + "\n"
                                    f"Сумма: " + str(np.sum(n_arr)) + "\n"
                                                                      f"n_i  - n*p_i: " + str(
        n_arr - p_arr * sz) + "\n"
                              f"res: " + str(res_arr) + "\n"
                                                        f"res_sum = " + str(np.sum(res_arr)) + "\n")

    rows = [[i + 1, ("%.2f" % borders[(i + 1) % (k - 1)]) + ', ' + "%.2f" % borders[(i + 2) % (k - 1)],
             "%.2f" % n_arr[i],
             "%.4f" % p_arr[i],
             "%.2f" % (sz * p_arr[i]),
             "%.2f" % (n_arr[i] - sz * p_arr[i]),
             "%.2f" % ((n_arr[i] - sz * p_arr[i]) ** 2 / (sz * p_arr[i]))] for i in range(k)]
    rows.insert(0, ['line i', 'Границы Delta_i', 'n_i', 'p_i', 'np_i' 'n_i - np_i',
                    '(n_i - np_i)^2/np_i'])
    with open("task7_data/task7_laplace.txt", "w") as f:
        for row in rows:
            f.write(" ".join([str(i) for i in row]) + "\n")


def task7():
    if not os.path.exists("task7_data"):
        os.mkdir("task7_data")
    print("Нормальное распределение")
    normal_exp()
    print("Распределение Лапласа")
    laplace_exp()

    gamma = 0.95

    alpha = 1 - gamma

    def get_student_mo(samples, alpha):
        med = np.mean(samples)
        n = len(samples)
        s = np.std(samples)
        t_a = t.ppf(1 - alpha / 2, n - 1)
        q_1 = med - s * t_a / np.sqrt(n - 1)
        q_2 = med + s * t_a / np.sqrt(n - 1)
        return q_1, q_2

    def get_chi_sigma(samples, alpha):
        n = len(samples)
        s = np.std(samples)
        q_1 = s * np.sqrt(n) / np.sqrt(chi2.ppf(1 - alpha / 2, n - 1))
        q_2 = s * np.sqrt(n) / np.sqrt(chi2.ppf(alpha / 2, n - 1))
        return q_1, q_2

    def get_as_mo(samples, alpha):
        med = np.mean(samples)
        n = len(samples)
        s = np.std(samples)
        u = norm.ppf(1 - alpha / 2)
        q_1 = med - s * u / np.sqrt(n)
        q_2 = med + s * u / np.sqrt(n)
        return q_1, q_2


def get_as_sigma(samples, alpha):
    med = np.mean(samples)
    n = len(samples)
    s = np.std(samples)
    u = norm.ppf(1 - alpha / 2)
    m4 = moment(samples, 4)
    e = m4 / (s * s * s * s)
    U = u * np.sqrt((e + 2) / n)
    q_1 = s / np.sqrt(1 + U)
    q_2 = s / np.sqrt(1 - U)
    return q_1, q_2


def task8():
    samples20 = np.random.normal(0, 1, size=20)
    samples100 = np.random.normal(0, 1, size=100)

    # classic interval params
    student_20 = get_student_mo(samples20, alpha)
    student_100 = get_student_mo(samples100, alpha)

    chi_20 = get_chi_sigma(samples20, alpha)
    chi_100 = get_chi_sigma(samples100, alpha)

    # assimptotic interval params
    as_mo_20 = get_as_mo(samples20, alpha)
    as_mo_100 = get_as_mo(samples100, alpha)

    as_d_20 = get_as_sigma(samples20, alpha)
    as_d_100 = get_as_sigma(samples100, alpha)

    print(f"Асимптотический подход:\n"
          f"n = 20 \n"
          f"\t\t m: " + str(as_mo_20) + " \t sigma: " + str(as_d_20) + "\n"
                                                                       f"n = 100 \n"
                                                                       f"\t\t m: " + str(
        as_mo_100) + " \t sigma: " + str(as_d_100) + "\n")

    print(f"Классический подход:\n"
          f"n = 20 \n"
          f"\t\t m: " + str(student_20) + " \t sigma: " + str(chi_20) + "\n"
                                                                        f"n = 100 \n"
                                                                        f"\t\t m: " + str(
        student_100) + " \t sigma: " + str(chi_100) + "\n")


if __name__ == "__main__":
   # task5()
    #task6()
    #task7()
    task8()