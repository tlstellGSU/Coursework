import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import brentq
from scipy.special import erfc
import scipy.stats as stats
from sklearn.neighbors import KernelDensity

# question 3

Q3_data_vec = [412, 380, 480, 385, 555, 360, 395, 370, 470, 382, 534, 371, 388, 355, 460, 379, 589, 285, 450, 600, 455, 365, 565, 366, 510, 590, 440, 350, 550, 382, 477, 575, 435, 345, 588, 379, 465, 560, 420, 340, 591, 355,430, 540, 410, 335, 583, 360, 421, 525, 405, 330, 578, 396, 399, 515, 390, 325, 595, 899]

# 3.a

def plot_histogram(data, bin_width, save_folder = None):
    data = np.array(data, dtype=float)

    # Build the edges explicitly so every bin is exactly bin_width wide.
    # Passing an integer count instead lets matplotlib pick width
    # range/count, which only equals bin_width when it divides the range.
    n_bins = int(np.ceil((np.max(data) - np.min(data)) / bin_width))
    edges = np.min(data) + bin_width * np.arange(n_bins + 1)

    plt.hist(data, bins=edges, edgecolor='black')
    plt.title(f'Histogram (bin width = {bin_width:g}, {n_bins} bins)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    #plt.show()

    if save_folder:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_folder / f"histogram_width{bin_width:g}.png")

    plt.close()

# 3.b

def kernel_distribution(data, bandwidth, save_folder = None, label = None):
    data = np.array(data, dtype=float).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data)

    # np.min/np.max, not the builtins: min() on an (n, 1) array returns a
    # 1-element array, which makes x_d 3-D and score_samples reject it.
    # Pad by 3 bandwidths so the tails decay instead of being clipped at
    # the data range.
    pad = 3 * bandwidth
    x_d = np.linspace(np.min(data) - pad, np.max(data) + pad, 1000)[:, np.newaxis]
    log_dens = kde.score_samples(x_d)
    dens = np.exp(log_dens)

    label_names = {
        "silverman": "Silverman's Rule",
        "sheather_jones": "Sheather-Jones (Plug-in)",
        "scott": "Scott's Rule",
        "silverman_no_outlier": "Silverman's Rule (No Outlier)",
        "sheather_jones_no_outlier": "Sheather-Jones (No Outlier)",
        "scott_no_outlier": "Scott's Rule (No Outlier)",
    }
    method_name = label_names.get(label, (label.replace('_', ' ').title() if label else "Gaussian KDE"))
    curve_label = f"{method_name} (h = {bandwidth:.3f})"

    plt.figure(figsize=(7, 5))
    plt.plot(x_d[:, 0], dens, color='steelblue', lw=2, label=curve_label)
    plt.fill_between(x_d[:, 0], dens, alpha=0.35, color='steelblue')
    plt.plot(data[:, 0], np.zeros_like(data[:, 0]), '|', color='black',
             markeredgewidth=1.2, markersize=8, label='Data points')

    plt.title(f'Kernel Density Estimation: {method_name} (h = {bandwidth:.3f})')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_folder:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_folder / f"kernel_density_{label or round(bandwidth, 3)}.png", dpi=150)

    plt.close()


def plot_combined_kernel_distributions(data, bandwidth_dict, save_folder = None,
                                       filename = "kernel_density_comparison.png",
                                       title = "Kernel Density Estimation Comparison"):
    """Plot multiple KDE curves with different bandwidths on the same axis with labels."""
    data = np.array(data, dtype=float).reshape(-1, 1)

    max_bw = max(bandwidth_dict.values())
    pad = 3 * max_bw
    x_d = np.linspace(np.min(data) - pad, np.max(data) + pad, 1000)[:, np.newaxis]

    plt.figure(figsize=(8, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    styles = ['-', '--', '-.', ':']

    label_names = {
        "silverman": "Silverman's Rule",
        "sheather_jones": "Sheather-Jones",
        "scott": "Scott's Rule",
    }

    for idx, (name, bw) in enumerate(bandwidth_dict.items()):
        kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(data)
        log_dens = kde.score_samples(x_d)
        dens = np.exp(log_dens)

        disp_name = label_names.get(name, name.replace('_', ' ').title())
        c = colors[idx % len(colors)]
        s = styles[idx % len(styles)]
        plt.plot(x_d[:, 0], dens, color=c, linestyle=s, lw=2,
                 label=f"{disp_name} (h = {bw:.3f})")
        plt.fill_between(x_d[:, 0], dens, color=c, alpha=0.08)

    plt.plot(data[:, 0], np.zeros_like(data[:, 0]), '|', color='black',
             markeredgewidth=1.2, markersize=8, label='Data points')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_folder:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_folder / filename, dpi=150)

    plt.close()


# 3.b / 3.c -- bandwidth selection rules


def _pair_diffs(x):
    """Absolute pairwise differences |x_i - x_j| for i < j."""
    d = np.abs(x[:, None] - x[None, :])
    return d[np.triu_indices_from(d, k=1)]


def _psi4(d, n, h):
    """Pilot estimate of Psi_4 = integral of f''(x)^2 (R calls this SDh)."""
    u2 = (d / h) ** 2
    s = np.sum(np.exp(-u2 / 2) * (u2 * u2 - 6 * u2 + 3))
    return (2 * s + 3 * n) / (n * (n - 1) * h ** 5 * np.sqrt(2 * np.pi))


def _psi6(d, n, h):
    """Pilot estimate of Psi_6 (R calls this TDh)."""
    u2 = (d / h) ** 2
    s = np.sum(np.exp(-u2 / 2) * (u2 ** 3 - 15 * u2 * u2 + 45 * u2 - 15))
    return (2 * s - 15 * n) / (n * (n - 1) * h ** 7 * np.sqrt(2 * np.pi))


def bw_silverman(data, show_steps = False):
    """Silverman's rule of thumb: h = 0.9 * A * n^(-1/5), A = min(sd, IQR/1.34).

    Silverman (1986), Density Estimation for Statistics and Data Analysis,
    eq. 3.31. show_steps=True prints the worked arithmetic for Q3.c.
    """
    x = np.asarray(data, dtype=float)
    n = x.size
    sd = x.std(ddof=1)
    q1, q3 = np.percentile(x, 25), np.percentile(x, 75)
    iqr = q3 - q1
    a_term = min(sd, iqr / 1.34)
    h = 0.9 * a_term * n ** (-1 / 5)

    if show_steps:
        print("Q3.c -- Silverman's rule of thumb, step by step")
        print(f"  1. n                       = {n}")
        print(f"  2. sample sd (ddof = 1)    = {sd:.4f}")
        print(f"  3. Q1 = {q1:.4f}, Q3 = {q3:.4f}  ->  IQR = {iqr:.4f}")
        print(f"  4. IQR / 1.34              = {iqr / 1.34:.4f}")
        print( "     (for a normal, IQR = 1.34*sigma, so IQR/1.34 is a")
        print( "      robust estimate of sigma that an outlier cannot inflate)")
        print(f"  5. A = min(sd, IQR/1.34)   = {a_term:.4f}"
              f"   <- {'sd' if sd <= iqr / 1.34 else 'IQR/1.34'} is the smaller")
        print(f"  6. n^(-1/5)                = {n ** (-1 / 5):.6f}")
        print(f"  7. h = 0.9 * {a_term:.4f} * {n ** (-1 / 5):.6f} = {h:.4f}")

    return h


def bw_scott(data):
    """Scott's rule -- Silverman's form with 1.06 in place of 0.9."""
    x = np.asarray(data, dtype=float)
    n = x.size
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    return 1.06 * min(x.std(ddof=1), iqr / 1.34) * n ** (-1 / 5)


def bw_sheather_jones(data, method = "ste"):
    """Sheather-Jones plug-in bandwidth (Sheather & Jones, JRSS-B 1991).

    method="ste" is the solve-the-equation rule (what R's bw.SJ does by
    default); "dpi" is the direct plug-in. The pairwise sums here are exact
    rather than binned, so this is marginally more accurate than bw.SJ's
    defaults.
    """
    x = np.asarray(data, dtype=float)
    n = x.size
    d = _pair_diffs(x)
    sd = x.std(ddof=1)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    scale = min(sd, iqr / 1.349)

    a = 1.24 * scale * n ** (-1 / 7)
    b = 1.23 * scale * n ** (-1 / 9)
    c1 = 1 / (2 * np.sqrt(np.pi) * n)

    td = -_psi6(d, n, b)
    if not np.isfinite(td) or td <= 0:
        raise ValueError("Sheather-Jones: sample too sparse for the pilot estimate")

    if method == "dpi":
        return (c1 / _psi4(d, n, (2.394 / (n * td)) ** (1 / 7))) ** (1 / 5)

    alpha2 = 1.357 * (_psi4(d, n, a) / td) ** (1 / 7)

    def objective(h):
        return (c1 / _psi4(d, n, alpha2 * h ** (5 / 7))) ** (1 / 5) - h

    hmax = 1.144 * sd * n ** (-1 / 5)
    lo, hi = 0.1 * hmax, hmax
    for i in range(100):
        if objective(lo) * objective(hi) < 0:
            return brentq(objective, lo, hi, xtol=1e-12)
        if i % 2:
            lo /= 1.2
        else:
            hi *= 1.2
    raise ValueError("Sheather-Jones: no root in the bandwidth search range")

# 3.d 

def QQ_plot(data, save_folder = None):
    data = np.asarray(data, dtype=float)
    plt.figure(figsize=(6, 5))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title('Normal Q-Q Plot')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_folder:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_folder / "qq_plot.png", dpi=150)

    plt.close()

# 3.e

def box_plot(data, save_folder = None):
    data = np.asarray(data, dtype=float)

    min_val = float(np.min(data))
    q1 = float(np.percentile(data, 25))
    median = float(np.median(data))
    q3 = float(np.percentile(data, 75))
    iqr = float(q3 - q1)
    lower_fence = float(q1 - 1.5 * iqr)
    upper_fence = float(q3 + 1.5 * iqr)
    outliers = [float(x) for x in data if x < lower_fence or x > upper_fence]

    plt.figure(figsize=(7, 6))
    bp = plt.boxplot(data, vert=True, patch_artist=True)

    # matplotlib draws the box from the same linear-interpolated quartiles
    # as np.percentile, so each label sits on the line it names. Connectors
    # start at the drawn box and whisker cap, whose width matplotlib picks.
    box_x = bp['boxes'][0].get_path().vertices[:, 0]
    cap_right = max(bp['caps'][0].get_xdata())
    label_x = box_x.max() + 0.15
    labels = [(cap_right, min_val, "Min"), (box_x.max(), q1, "Q1"),
              (box_x.max(), median, "Median"), (box_x.max(), q3, "Q3")]
    labels += [(1.02, y, "Outlier") for y in outliers]
    for x_anchor, y, name in labels:
        plt.annotate(f"{name} = {y:g}", xy=(x_anchor, y), xytext=(label_x, y),
                     va='center', fontsize=9,
                     arrowprops=dict(arrowstyle='-', color='gray', lw=0.8,
                                     shrinkA=0, shrinkB=0))
    bracket_x = box_x.min() - 0.06
    plt.annotate('', xy=(bracket_x, q1), xytext=(bracket_x, q3),
                 arrowprops=dict(arrowstyle='<->', color='black', lw=1,
                                 shrinkA=0, shrinkB=0))
    plt.text(bracket_x - 0.015, (q1 + q3) / 2, f"IQR = {iqr:g}", rotation=90,
             ha='right', va='center', fontsize=9)
    plt.text(0.02, 0.98, f"1.5*IQR fences: [{lower_fence:g}, {upper_fence:g}]",
             transform=plt.gca().transAxes, ha='left', va='top', fontsize=9)

    plt.xlim(0.65, 1.45)
    plt.xticks([1], ['Sample Data'])
    plt.title('Box Plot')
    plt.ylabel('Value')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_folder:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_folder / "box_plot.png", dpi=150)

    plt.close()

    return {
        "min": min_val,
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "outliers": outliers,
    }


def peirce_ratio(N, n, m = 1):
    """Peirce's ratio x = max allowed |x_i - mean| / sd (Ross 2003 calls it R).

    N observations, n of them doubtful, m fitted unknowns (1 for a mean).
    Gould's (1855) equations as implemented in the reference code on
    Wikipedia's "Peirce's criterion" page, iterating on the probability
    ratio r until it converges:
        Q^N            = n^n (N - n)^(N - n) / N^N
        lambda^(N - n) = Q^N / r^n
        x^2            = 1 + (N - m - n) / n * (1 - lambda^2)
        r              = exp((x^2 - 1) / 2) * erfc(x / sqrt(2))
    Checked against every entry of Ross's table: the largest difference for
    4 <= N <= 60 is 0.0009 (N = 60, n = 1: 2.6623 here, 2.663 in the table).
    The one exception is N = 3 (1.2163 here, 1.196 in the table).
    """
    N, n, m = float(N), float(n), float(m)
    Q = n ** (n / N) * (N - n) ** ((N - n) / N) / N
    r_new, r_old = 1.0, 0.0
    x2 = 0.0
    while abs(r_new - r_old) > N * 2.0e-16:
        lam = (Q ** N / (r_new ** n or 1.0e-6)) ** (1.0 / (N - n))
        x2 = 1.0 + (N - m - n) / n * (1.0 - lam ** 2)
        if x2 < 0:
            x2, r_old = 0.0, r_new
        else:
            r_old, r_new = r_new, np.exp((x2 - 1.0) / 2.0) * erfc(np.sqrt(x2 / 2.0))
    return float(np.sqrt(x2))


def peirce_outliers(data, m = 1):
    """Peirce's criterion, following steps 1-9 of Ross (2003).

    S. M. Ross, "Peirce's criterion for the elimination of suspect
    experimental data", Journal of Engineering Technology, Fall 2003.
    The mean, the sample sd (ddof = 1) and N stay at their full-sample
    values throughout; only the assumed number of doubtful observations n
    changes. Start at n = 1; if k points have |x - mean| > R * sd, retry with
    n = k + 1; stop once no further point is rejected, then recompute the
    mean and sd without the rejected points. Assumes normal errors.
    """
    x = np.asarray(data, dtype=float)
    N = x.size
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    dev = np.abs(x - mean)

    rejected = np.zeros(N, dtype=bool)
    steps = []
    n = 1
    while n < N - m:
        ratio = peirce_ratio(N, n, m)
        flagged = dev > ratio * sd
        steps.append({"n": n, "R": ratio, "max_dev": ratio * sd,
                      "flagged": sorted(float(v) for v in x[flagged])})
        # R falls as n grows, so flagged always contains rejected.
        if flagged.sum() <= rejected.sum():
            break
        rejected = flagged
        n = int(rejected.sum()) + 1

    kept = x[~rejected]
    return {
        "N": N,
        "mean": mean,
        "sd": sd,
        "steps": steps,
        "outliers": sorted(float(v) for v in x[rejected]),
        "largest_kept_dev": float(dev[~rejected].max()),
        "kept_N": int(kept.size),
        "kept_mean": float(kept.mean()),
        "kept_sd": float(kept.std(ddof=1)),
    }


def violin_plot(data, save_folder = None, label = None):
    """Generate and save a standalone violin plot with median, mean, and data points."""
    data = np.asarray(data, dtype=float)
    plt.figure(figsize=(6, 5))

    vp = plt.violinplot(data, vert=True, showmeans=True, showmedians=True, showextrema=True)
    for pc in vp['bodies']:
        pc.set_facecolor('#2ca02c')
        pc.set_edgecolor('#1b611b')
        pc.set_alpha(0.6)
    vp['cmedians'].set_color('darkred')
    vp['cmedians'].set_linewidth(2)
    vp['cmeans'].set_color('navy')
    vp['cmeans'].set_linestyle('--')
    vp['cmeans'].set_linewidth(1.5)
    vp['cbars'].set_color('black')
    vp['cmins'].set_color('black')
    vp['cmaxes'].set_color('black')

    # Rug plot / jitter points
    rng = np.random.default_rng(42)
    jitter = rng.normal(1.0, 0.035, size=len(data))
    plt.scatter(jitter, data, alpha=0.45, color='black', s=20, label='Data points')

    title = f"Violin Plot{' (' + label.replace('_', ' ').title() + ')' if label else ''}"
    plt.title(title)
    plt.ylabel('Value')
    plt.xticks([1], ['Sample Data'])
    plt.grid(True, linestyle='--', alpha=0.5)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='darkred', lw=2, label='Median'),
        Line2D([0], [0], color='navy', lw=1.5, ls='--', label='Mean'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', alpha=0.5, markersize=6, label='Data points'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()

    if save_folder:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        fname = f"violin_plot_{label}.png" if label else "violin_plot.png"
        plt.savefig(save_folder / fname, dpi=150)

    plt.close()


def box_and_violin_plot(data, save_folder = None, label = None):
    """Generate and save a side-by-side box plot and violin plot comparison."""
    data = np.asarray(data, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # Left: Box Plot
    axes[0].boxplot(
        data, vert=True, patch_artist=True,
        boxprops=dict(facecolor='#aec7e8', edgecolor='#1f77b4', lw=1.5),
        medianprops=dict(color='darkred', lw=2),
        whiskerprops=dict(color='black', lw=1.2),
        capprops=dict(color='black', lw=1.2),
        flierprops=dict(marker='o', markerfacecolor='crimson', markeredgecolor='black', markersize=7)
    )
    axes[0].set_title('Box Plot')
    axes[0].set_ylabel('Value')
    axes[0].set_xticks([1])
    axes[0].set_xticklabels(['Sample Data'])
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Right: Violin Plot
    vp = axes[1].violinplot(data, vert=True, showmeans=True, showmedians=True, showextrema=True)
    for pc in vp['bodies']:
        pc.set_facecolor('#98df8a')
        pc.set_edgecolor('#2ca02c')
        pc.set_alpha(0.7)
    vp['cmedians'].set_color('darkred')
    vp['cmedians'].set_linewidth(2)
    vp['cmeans'].set_color('navy')
    vp['cmeans'].set_linestyle('--')
    vp['cmeans'].set_linewidth(1.5)
    vp['cbars'].set_color('black')
    vp['cmins'].set_color('black')
    vp['cmaxes'].set_color('black')

    # Overlay jittered data points
    rng = np.random.default_rng(42)
    jitter = rng.normal(1.0, 0.035, size=len(data))
    axes[1].scatter(jitter, data, alpha=0.45, color='black', s=20, label='Data points')

    axes[1].set_title('Violin Plot')
    axes[1].set_xticks([1])
    axes[1].set_xticklabels(['Sample Data'])
    axes[1].grid(True, linestyle='--', alpha=0.5)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='darkred', lw=2, label='Median'),
        Line2D([0], [0], color='navy', lw=1.5, ls='--', label='Mean'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', alpha=0.5, markersize=6, label='Data points'),
    ]
    axes[1].legend(handles=legend_elements, loc='upper right')

    title_suffix = f" ({label.replace('_', ' ').title()})" if label else ""
    fig.suptitle(f'Distribution Comparison: Box Plot vs. Violin Plot{title_suffix}', fontsize=13)
    plt.tight_layout()

    if save_folder:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        fname = f"box_and_violin_plot_{label}.png" if label else "box_and_violin_plot.png"
        fig.savefig(save_folder / fname, dpi=150)

    plt.close(fig)

# question 4

Q4_p1 = np.array([22, 1, 42, 10], dtype=float)
Q4_p2 = np.array([20, 0, 36, 8], dtype=float)

def compute_minkowski_distance(p1, p2, p=2):
    """Minkowski distance of order p between two vectors."""
    return float(np.sum(np.abs(np.asarray(p1) - np.asarray(p2)) ** p) ** (1 / p))

def compute_supremum_distance(p1, p2):
    """Supremum (Chebyshev / L_infinity) distance between two vectors."""
    return float(np.max(np.abs(np.asarray(p1) - np.asarray(p2))))

# question 5

Q5_data_df = pd.DataFrame(
    [
        [0.661, 1.454, 0.666, 0.912],
        [0.725, 1.256, 0.589, 1.024],
        [0.654, 1.412, 0.758, 1.102],
        [0.632, 1.896, 0.729, 0.987],
        [0.837, 1.522, 0.841, 0.923],
    ],
    index=["X1", "X2", "X3", "X4", "X5"],
    columns=["Dim 1", "Dim 2", "Dim 3", "Dim 4"],
)

Q5_query_x = np.array([0.78, 1.6, 0.7, 1.1])

# 5.a

def compute_euclidean_distance(df, query=None):
    if query is not None:
        query = np.asarray(query, dtype=float)
        return np.linalg.norm(df.values - query, axis=1)
    distances = np.zeros((df.shape[0], df.shape[0]))
    for i in range(df.shape[0]):
        for j in range(df.shape[0]):
            distances[i, j] = np.linalg.norm(df.iloc[i] - df.iloc[j])
    return distances

def compute_correlation_distance(df, query=None):
    if query is not None:
        query = np.asarray(query, dtype=float)
        corrs = np.array([np.corrcoef(df.iloc[i], query)[0, 1] for i in range(len(df))])
        return 1 - corrs
    # Sample-wise correlation distance (1 - r) across data rows
    correlation_matrix = df.T.corr()
    correlation_distance = 1 - correlation_matrix
    return correlation_distance.values

def compute_cosine_distance(df, query=None):
    if query is not None:
        query = np.asarray(query, dtype=float)
        norm_df = np.linalg.norm(df.values, axis=1)
        norm_q = np.linalg.norm(query)
        cosine_sim = np.dot(df.values, query) / (norm_df * norm_q)
        return 1 - cosine_sim
    cosine_similarity = np.dot(df, df.T) / (np.linalg.norm(df, axis=1)[:, np.newaxis] * np.linalg.norm(df, axis=1))
    cosine_distance = 1 - cosine_similarity
    return cosine_distance

def compute_dot_product_distance(df, query=None):
    if query is not None:
        query = np.asarray(query, dtype=float)
        return np.dot(df.values, query)
    dot_product_distance = np.zeros((df.shape[0], df.shape[0]))
    for i in range(df.shape[0]):
        for j in range(df.shape[0]):
            dot_product_distance[i, j] = np.dot(df.iloc[i], df.iloc[j])
    return dot_product_distance

def rank_query_similarities(df, query):
    """Compute similarities and rank database points against a query vector for Q5.a."""
    query = np.asarray(query, dtype=float)

    euc_dist = compute_euclidean_distance(df, query=query)
    corr_dist = compute_correlation_distance(df, query=query)
    corr_sim = 1 - corr_dist
    cos_dist = compute_cosine_distance(df, query=query)
    cos_sim = 1 - cos_dist
    dot_prod = compute_dot_product_distance(df, query=query)

    summary_df = pd.DataFrame({
        "Euclidean Dist": euc_dist,
        "Correlation": corr_sim,
        "Cosine Sim": cos_sim,
        "Dot Product": dot_prod,
    }, index=df.index)

    rankings_df = pd.DataFrame({
        "Euclidean (min dist)": summary_df["Euclidean Dist"].sort_values(ascending=True).index.tolist(),
        "Correlation (max corr)": summary_df["Correlation"].sort_values(ascending=False).index.tolist(),
        "Cosine Sim (max cos)": summary_df["Cosine Sim"].sort_values(ascending=False).index.tolist(),
        "Dot Product (max dot)": summary_df["Dot Product"].sort_values(ascending=False).index.tolist(),
    }, index=[f"Rank {i+1}" for i in range(len(df))])

    return summary_df, rankings_df


def plot_question_5_rankings(summary_df, query, save_folder = None):
    """Generate and save a 4-panel bar chart comparing similarity measures and rankings for Q5.a."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    metrics = [
        ("Euclidean Dist", axes[0, 0], "Euclidean Distance (Smaller = More Similar)", "#1f77b4", True),
        ("Correlation", axes[0, 1], "Correlation (Larger = More Similar)", "#ff7f0e", False),
        ("Cosine Sim", axes[1, 0], "Cosine Similarity (Larger = More Similar)", "#2ca02c", False),
        ("Dot Product", axes[1, 1], "Dot Product (Larger = More Similar)", "#d62728", False),
    ]

    for col, ax, title, color, asc in metrics:
        sorted_s = summary_df[col].sort_values(ascending=asc)
        bars = ax.bar(sorted_s.index, sorted_s.values, color=color, edgecolor="black", alpha=0.8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    query_str = f"({', '.join(str(v) for v in np.asarray(query).tolist())})"
    fig.suptitle(f"Question 5.a: Database Point Comparison Against Query x = {query_str}", fontsize=13)
    plt.tight_layout()

    if save_folder:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_folder / "question_5_similarity_comparison.png", dpi=150)

    plt.close(fig)

def main():
    # Relative to this file, so the script runs from any working directory.
    output_folder_path = Path(__file__).resolve().parent / "HW_1_output"
    output_folder_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CSC 6740 - Homework 1 Solution Outputs")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Question 3.a -- Histograms
    # -------------------------------------------------------------------------
    print("\n[Question 3.a] Histograms with bin widths [10, 50, 100]")
    bin_widths = [10, 50, 100]
    for bin_width in bin_widths:
        plot_histogram(Q3_data_vec, bin_width, save_folder=output_folder_path)
    print(f"  Histograms saved to: {output_folder_path / 'histogram_width*.png'}")

    # -------------------------------------------------------------------------
    # Question 3.b -- Kernel Density Estimation
    # -------------------------------------------------------------------------
    bandwidths = {
        "silverman": bw_silverman(Q3_data_vec),
        "sheather_jones": bw_sheather_jones(Q3_data_vec),
        "scott": bw_scott(Q3_data_vec),
    }
    print("\n[Question 3.b] Kernel Density Estimation Bandwidths:")
    for name, bandwidth in bandwidths.items():
        print(f"  {name:15s} h = {bandwidth:.4f}")
        kernel_distribution(Q3_data_vec, bandwidth,
                            save_folder=output_folder_path, label=name)
    plot_combined_kernel_distributions(
        Q3_data_vec,
        bandwidths,
        save_folder=output_folder_path,
        filename="kernel_density_comparison.png",
        title="Kernel Density Estimation Comparison (Silverman, SJ, Scott)"
    )
    print(f"  Individual KDE plots saved to: {output_folder_path / 'kernel_density_*.png'}")
    print(f"  Combined KDE comparison plot saved to: {output_folder_path / 'kernel_density_comparison.png'}")

    # -------------------------------------------------------------------------
    # Question 3.c -- Silverman's manual step-by-step arithmetic
    # -------------------------------------------------------------------------
    print("\n[Question 3.c] Manual calculation steps for Silverman's bandwidth:")
    bw_silverman(Q3_data_vec, show_steps=True)

    # -------------------------------------------------------------------------
    # Question 3.d -- Normal Q-Q Plot
    # -------------------------------------------------------------------------
    print("\n[Question 3.d] Generating Normal Q-Q Plot...")
    QQ_plot(Q3_data_vec, save_folder=output_folder_path)
    print(f"  Q-Q plot saved to: {output_folder_path / 'qq_plot.png'}")

    # -------------------------------------------------------------------------
    # Question 3.e -- Box Plot, Violin Plot & Outlier Analysis
    # -------------------------------------------------------------------------
    print("\n[Question 3.e] Box Plot, Violin Plot, and Outlier Detection (1.5 * IQR rule, Peirce's criterion):")
    box_stats = box_plot(Q3_data_vec, save_folder=output_folder_path)
    peirce = peirce_outliers(Q3_data_vec)
    violin_plot(Q3_data_vec, save_folder=output_folder_path)
    box_and_violin_plot(Q3_data_vec, save_folder=output_folder_path)
    print(f"  Box plot saved to:               {output_folder_path / 'box_plot.png'}")
    print(f"  Violin plot saved to:            {output_folder_path / 'violin_plot.png'}")
    print(f"  Side-by-side comparison saved to:{output_folder_path / 'box_and_violin_plot.png'}")

    # Hand-calculation convention: Q1 and Q3 as the medians of the lower and
    # upper halves of the sorted data.
    sorted_q3 = np.sort(np.asarray(Q3_data_vec, dtype=float))
    half = sorted_q3.size // 2
    hand_q1 = float(np.median(sorted_q3[:half]))
    hand_q3 = float(np.median(sorted_q3[-half:]))
    hand_iqr = hand_q3 - hand_q1
    hand_outliers = [float(x) for x in Q3_data_vec
                     if x < hand_q1 - 1.5 * hand_iqr or x > hand_q3 + 1.5 * hand_iqr]

    q3e_lines = [
        "Box plot summary (quartiles by linear interpolation, the numpy/matplotlib default):",
        f"  Minimum:                  {box_stats['min']:.4f}",
        f"  First Quartile (Q1):      {box_stats['q1']:.4f}",
        f"  Median:                   {box_stats['median']:.4f}",
        f"  Third Quartile (Q3):      {box_stats['q3']:.4f}",
        f"  IQR (Q3 - Q1):            {box_stats['iqr']:.4f}",
        f"  Lower Fence (Q1-1.5*IQR): {box_stats['lower_fence']:.4f}",
        f"  Upper Fence (Q3+1.5*IQR): {box_stats['upper_fence']:.4f}",
        f"  Outliers (1.5*IQR rule):  {box_stats['outliers']}",
        f"  By hand (median of each half): Q1 = {hand_q1:g}, Q3 = {hand_q3:g}, "
        f"IQR = {hand_iqr:g}, 1.5*IQR outliers {hand_outliers}",
        "",
        "Peirce's criterion (Ross 2003; mean, sample sd and N kept at full-sample values):",
        f"  N = {peirce['N']}, mean = {peirce['mean']:.4f}, sample sd = {peirce['sd']:.4f}",
    ]
    for step in peirce["steps"]:
        q3e_lines.append(
            f"  n = {step['n']} doubtful: R = {step['R']:.4f}, "
            f"max |x - mean| = R * sd = {step['max_dev']:.4f}, beyond it: {step['flagged']}")
    q3e_lines += [
        f"  Outliers (Peirce):        {peirce['outliers']}",
        f"  Largest retained |x - mean|: {peirce['largest_kept_dev']:.4f}",
        f"  Without outliers: N = {peirce['kept_N']}, mean = {peirce['kept_mean']:.4f}, "
        f"sample sd = {peirce['kept_sd']:.4f}",
        f"  1.5*IQR rule and Peirce's criterion agree: "
        f"{sorted(box_stats['outliers']) == peirce['outliers']}",
    ]
    for line in q3e_lines:
        print("  " + line)

    q3e_txt_file = output_folder_path / "question_3e_output.txt"
    with open(q3e_txt_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("CSC 6740 - Question 3.e Outputs\n")
        f.write("=" * 70 + "\n\n")
        f.write("\n".join(q3e_lines) + "\n")
    print(f"  Text report saved to: {q3e_txt_file}")

    # -------------------------------------------------------------------------
    # Question 3.f -- Repeat step b without outliers
    # -------------------------------------------------------------------------
    outliers_set = set(box_stats["outliers"])
    q3_clean = [x for x in Q3_data_vec if x not in outliers_set]
    print(f"\n[Question 3.f] Repeating step b after removing outlier(s) {box_stats['outliers']}:")
    print(f"  Original sample size: {len(Q3_data_vec)}, Cleaned sample size: {len(q3_clean)}")
    bw_clean = {
        "silverman": bw_silverman(q3_clean),
        "sheather_jones": bw_sheather_jones(q3_clean),
        "scott": bw_scott(q3_clean),
    }
    for name, bandwidth in bw_clean.items():
        print(f"  {name:15s} h = {bandwidth:.4f}")
        kernel_distribution(q3_clean, bandwidth,
                            save_folder=output_folder_path, label=f"{name}_no_outlier")
    plot_combined_kernel_distributions(
        q3_clean,
        bw_clean,
        save_folder=output_folder_path,
        filename="kernel_density_comparison_no_outlier.png",
        title="Kernel Density Estimation Comparison - No Outlier (Silverman, SJ, Scott)"
    )
    violin_plot(q3_clean, save_folder=output_folder_path, label="no_outlier")
    box_and_violin_plot(q3_clean, save_folder=output_folder_path, label="no_outlier")
    print(f"  Cleaned KDE plots saved to: {output_folder_path / 'kernel_density_*_no_outlier.png'}")
    print(f"  Cleaned combined KDE comparison plot saved to: {output_folder_path / 'kernel_density_comparison_no_outlier.png'}")
    print(f"  Cleaned violin plot saved to: {output_folder_path / 'violin_plot_no_outlier.png'}")
    print(f"  Cleaned box & violin comparison saved to: {output_folder_path / 'box_and_violin_plot_no_outlier.png'}")

    # -------------------------------------------------------------------------
    # Question 4 -- Distance measures between tuples
    # -------------------------------------------------------------------------
    print("\n[Question 4] Distance measures between (22, 1, 42, 10) and (20, 0, 36, 8):")
    euc_q4 = compute_minkowski_distance(Q4_p1, Q4_p2, p=2)
    man_q4 = compute_minkowski_distance(Q4_p1, Q4_p2, p=1)
    mink_3_q4 = compute_minkowski_distance(Q4_p1, Q4_p2, p=3)
    sup_q4 = compute_supremum_distance(Q4_p1, Q4_p2)
    print(f"  (a) Euclidean distance:            {euc_q4:.4f}")
    print(f"  (b) Manhattan distance:            {man_q4:.4f}")
    print(f"  (c) Minkowski distance (h = 3):    {mink_3_q4:.4f}")
    print(f"  (d) Supremum distance (L_inf):     {sup_q4:.4f}")

    # -------------------------------------------------------------------------
    # Question 5 -- Pairwise Distance / Similarity Matrices
    # -------------------------------------------------------------------------
    print("\n[Question 5] Pairwise Matrices for Database Objects (X1 - X5):")
    sample_labels = Q5_data_df.index.tolist()

    print("\nPairwise Euclidean Distance Matrix:")
    euc_df = pd.DataFrame(compute_euclidean_distance(Q5_data_df),
                          index=sample_labels, columns=sample_labels)
    print(euc_df.round(4).to_string())

    print("\nPairwise Correlation Distance Matrix (1 - r):")
    corr_df = pd.DataFrame(compute_correlation_distance(Q5_data_df),
                           index=sample_labels, columns=sample_labels)
    print(corr_df.round(4).to_string())

    print("\nPairwise Cosine Distance Matrix (1 - cosine_similarity):")
    cos_df = pd.DataFrame(compute_cosine_distance(Q5_data_df),
                          index=sample_labels, columns=sample_labels)
    print(cos_df.round(4).to_string())

    print("\nPairwise Dot Product Matrix:")
    dot_df = pd.DataFrame(compute_dot_product_distance(Q5_data_df),
                          index=sample_labels, columns=sample_labels)
    print(dot_df.round(4).to_string())

    # -------------------------------------------------------------------------
    # Question 5.a -- Query point similarity and ranking
    # -------------------------------------------------------------------------
    print(f"\n[Question 5.a] Ranking Database Points Against Query Point x = {Q5_query_x.tolist()}:")
    summary_q5a, rankings_q5a = rank_query_similarities(Q5_data_df, Q5_query_x)

    print("\nMetric Values Against Query:")
    print(summary_q5a.round(4).to_string())

    print("\nRankings (1st = Most Similar):")
    print(rankings_q5a.to_string())

    # Generate Question 5.a similarity plot
    plot_question_5_rankings(summary_q5a, Q5_query_x, save_folder=output_folder_path)

    # Save Question 5 tables and text output to HW_1_output/
    q5_txt_file = output_folder_path / "question_5_output.txt"
    with open(q5_txt_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("CSC 6740 - Question 5 Outputs\n")
        f.write("=" * 70 + "\n\n")
        f.write("1. Pairwise Euclidean Distance Matrix:\n")
        f.write(euc_df.round(4).to_string() + "\n\n")
        f.write("2. Pairwise Correlation Distance Matrix (1 - r):\n")
        f.write(corr_df.round(4).to_string() + "\n\n")
        f.write("3. Pairwise Cosine Distance Matrix (1 - cosine_similarity):\n")
        f.write(cos_df.round(4).to_string() + "\n\n")
        f.write("4. Pairwise Dot Product Matrix:\n")
        f.write(dot_df.round(4).to_string() + "\n\n")
        f.write("-" * 70 + "\n")
        f.write(f"Question 5.a: Query Point x = {Q5_query_x.tolist()}\n")
        f.write("-" * 70 + "\n\n")
        f.write("Similarity / Distance Values Against Query:\n")
        f.write(summary_q5a.round(4).to_string() + "\n\n")
        f.write("Rankings (1st = Most Similar):\n")
        f.write(rankings_q5a.to_string() + "\n")

    summary_q5a.round(4).to_csv(output_folder_path / "question_5_metrics.csv")
    rankings_q5a.to_csv(output_folder_path / "question_5_rankings.csv")
    euc_df.round(4).to_csv(output_folder_path / "question_5_pairwise_euclidean.csv")
    corr_df.round(4).to_csv(output_folder_path / "question_5_pairwise_correlation.csv")
    cos_df.round(4).to_csv(output_folder_path / "question_5_pairwise_cosine.csv")
    dot_df.round(4).to_csv(output_folder_path / "question_5_pairwise_dot_product.csv")

    print("\n" + "-" * 70)
    print("Question 5 files saved to:")
    print(f"  Text report: {output_folder_path / 'question_5_output.txt'}")
    print(f"  Plot:        {output_folder_path / 'question_5_similarity_comparison.png'}")
    print(f"  CSVs:        {output_folder_path / 'question_5_*.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()