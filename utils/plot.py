import matplotlib.pyplot as plt
from utils.metric import uplift_curve, qini_curve_industry


# =========================================================
# Plot uplift / gain curve
# =========================================================

def plot_uplift_curve(
    tau_hat,
    t,
    y,
    perfect_tau=None,
    show_perfect=True,
    show_random=True,
    figsize=(6, 5),
):
    """
    Plot industry-style uplift gain curve.
    """

    p, gain = uplift_curve(tau_hat, t, y)

    plt.figure(figsize=figsize)

    # model
    plt.plot(p,gain,linewidth=2,label="Model")

    # random baseline
    if show_random:
        plt.plot(
            p,
            gain[-1] * p,
            "--",
            linewidth=2,
            label="Random",
        )

    # oracle / pseudo-oracle
    if show_perfect:
        if perfect_tau is None:
            perfect_tau = y * (2 * t - 1)

        p_perf, gain_perf = uplift_curve(
            perfect_tau,
            t,
            y,
        )

        plt.plot(
            p_perf,
            gain_perf,
            linewidth=2,
            label="Oracle",
        )

    plt.xlabel("Population Share")
    plt.ylabel("Cumulative Gain")
    plt.title("Uplift Curve")

    plt.grid(True)
    plt.legend()

    plt.show()


# =========================================================
# Plot Qini curve
# =========================================================

def plot_qini_curve(
    tau_hat,
    t,
    y,
    perfect_tau=None,
    show_perfect=True,
    show_random=True,
    figsize=(6, 5),
):
    """
    Plot Qini curve.
    """

    p, qini = qini_curve_industry(tau_hat, t, y)

    plt.figure(figsize=figsize)

    # model
    plt.plot(
        p,
        qini,
        linewidth=2,
        label="Model",
    )

    # random baseline
    if show_random:
        plt.plot(
            p,
            qini[-1] * p,
            "--",
            linewidth=2,
            label="Random",
        )

    # oracle
    if show_perfect:

        if perfect_tau is None:
            perfect_tau = y * (2 * t - 1)

        p_perf, qini_perf = qini_curve_industry(
            perfect_tau,
            t,
            y,
        )

        plt.plot(
            p_perf,
            qini_perf,
            linewidth=2,
            label="Oracle",
        )

    plt.xlabel("Population Share")
    plt.ylabel("Qini Gain")
    plt.title("Qini Curve")

    plt.grid(True)
    plt.legend()

    plt.show()
