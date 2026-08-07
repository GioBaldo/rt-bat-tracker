# %% backend
# %matplotlib qt
# %% IMPORTS
import sim_localisation_mpr2003 as mpr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Simulate array shapes and computes the localization error with respect to a TDOA error
    This relates the array shape to its sensitivity to fainth errors on the TDOA which can be due to incorrect positioning of microphones or inaccuracy in computing delays 

    Reference coordinate system is (0, 0, 0) at the center of the array, with the z-axis pointing outwards. The array is assumed to be in the xy-plane at z=0.
"""
# %% FUNCTIONS


def StarArray(n_mic, sep, depth):
    """
    Create a star-shaped array of microphones.
    """
    angles = np.linspace(0, 2 * np.pi, n_mic - 1, endpoint=False)
    x = sep * np.cos(angles)
    y = sep * np.sin(angles)
    z = np.zeros(n_mic - 1)
    return np.vstack(((0, 0, depth), np.column_stack((x, y, z))))


def PrismArray(n_mic, sep, depth):
    """
    Create a prism-shaped array of microphones.
    """
    root = [
        (0, 0, 0),
        (0, sep, 0),
        (sep, 0, 0),
        (sep / 2, sep / 2, depth),
        (sep, sep, 0),
        (-sep, 0, 0),
        (-sep / 2, sep / 2, depth),
        (-sep, sep, 0),
    ]
    return np.array(root[:n_mic])


def ParaboloidArray(n_mic, sep, depth):
    """
    Create a paraboloid-shaped array of microphones.
    """
    pass


def RandomArray(n_mic, sep, depth):
    """
    Create a random-shaped array of microphones.
    """
    pass


def get_array(shape, n_mic, sep, depth):
    if shape == "star":
        array = StarArray(n_mic, sep, depth)
    elif shape == "prism":
        array = PrismArray(n_mic, sep, depth)
    elif shape == "paraboloid":
        array = ParaboloidArray(n_mic, sep, depth)
    elif shape == "random":
        array = RandomArray(n_mic, sep, depth)
    else:
        raise ValueError(f"Unknown array shape: {shape}")
    return array


def plot_arrays(sel):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    selected_arrays = sel["arr_id"].unique()

    for id in selected_arrays:
        arr_sel = sel[sel["arr_id"] == id]
        color = np.random.rand(3)
        ax.scatter(
            arr_sel["x"],
            arr_sel["y"],
            arr_sel["z"],
            color=color,
            alpha=0.8,
            s=50,
            label=f"array {id} - {arr_sel['shape'].iloc[0]} [{arr_sel['n_mic'].iloc[0]} mic / {arr_sel['sep'].iloc[0]} sep / {arr_sel['depth'].iloc[0]} depth]",
        )
        for _, row in arr_sel.iterrows():
            ax.text(row["x"], row["y"], row["z"], (str(id) + "." + str(row["mic_id"])))
            ax.plot(
                [0, row["x"]],
                [0, row["y"]],
                [0, row["z"]],
                color=color,
                alpha=0.5,
            )
    ax.set_title(f"array shapes {selected_arrays}")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((2, 2, 2))

    plt.show()


def plot_sources(sources_df, type="src", array_df=None, id=1):
    """
    Plot sources accordingly to the type specified [default:src]

    type:
    - src: plot the sources as they are
    - pure: plot the sources localized with pure delays
    - noisy: plot the sources localized with noisy delays
    - pure_c: plot the sources as they are, color-coded by pure localization error
    - noisy_c: plot the sources as they are, color-coded by noisy localization error
    - s+p: plot the sources as they are, and the sources localized with pure delays
    - s+n: plot the sources as they are, and the sources localized with noisy delays
    """
    sources_fig = plt.figure()
    sources_ax = sources_fig.add_subplot(111, projection="3d")
    sources_ax.scatter(0, 0, 0, color="red", s=20)
    src_radius = sources_df["r"].unique()
    pSIZE = 10
    pALPHA = 0.5
    match type:
        case "src":
            for r in src_radius:
                eq_r_df = sources_df[sources_df["r"] == r]
                sources_ax.scatter(
                    eq_r_df["x"],
                    eq_r_df["y"],
                    eq_r_df["z"],
                    s=pSIZE,
                    alpha=pALPHA,
                    label=f"r={r:.2f} m",
                )
        case "pure":
            for r in src_radius:
                eq_r_df = sources_df[sources_df["r"] == r]
                sources_ax.scatter(
                    eq_r_df["pl_x"],
                    eq_r_df["pl_y"],
                    eq_r_df["pl_z"],
                    s=pSIZE,
                    alpha=pALPHA,
                    label=f"r={r:.2f} m",
                )
        case "noisy":
            for r in src_radius:
                eq_r_df = sources_df[sources_df["r"] == r]
                sources_ax.scatter(
                    eq_r_df["nl_x"],
                    eq_r_df["nl_y"],
                    eq_r_df["nl_z"],
                    s=pSIZE,
                    alpha=pALPHA,
                    label=f"r={r:.2f} m",
                )
        case "pure_c":
            sources_ax.scatter(
                sources_df["x"],
                sources_df["y"],
                sources_df["z"],
                s=pSIZE,
                alpha=pALPHA,
                c=sources_df["pure_error"],
                cmap="viridis",
            )
        case "noisy_c":
            sources_ax.scatter(
                sources_df["x"],
                sources_df["y"],
                sources_df["z"],
                s=pSIZE,
                alpha=pALPHA,
                c=sources_df["noisy_error"],
                cmap="viridis",
            )
        case "s+p":
            sources_ax.scatter(
                sources_df["x"],
                sources_df["y"],
                sources_df["z"],
                s=pSIZE,
                alpha=pALPHA,
                label="sources",
            )
            sources_ax.scatter(
                sources_df["pl_x"],
                sources_df["pl_y"],
                sources_df["pl_z"],
                s=pSIZE,
                alpha=pALPHA,
                label="pure localized",
            )
            for src in sources_df.itertuples():
                sources_ax.plot(
                    [src.x, src.pl_x],
                    [src.y, src.pl_y],
                    [src.z, src.pl_z],
                    color="gray",
                    alpha=0.3,
                )
        case "s+n":
            sources_ax.scatter(
                sources_df["x"],
                sources_df["y"],
                sources_df["z"],
                s=pSIZE,
                alpha=pALPHA,
                label="sources",
            )
            sources_ax.scatter(
                sources_df["nl_x"],
                sources_df["nl_y"],
                sources_df["nl_z"],
                s=pSIZE,
                alpha=pALPHA,
                label="noisy localized",
            )
            for src in sources_df.itertuples():
                sources_ax.plot(
                    [src.x, src.nl_x],
                    [src.y, src.nl_y],
                    [src.z, src.nl_z],
                    color="gray",
                    alpha=0.3,
                )
    if array_df is not None:
        sel = array_df[array_df["arr_id"] == id]  # select here arrays to be evaluated
        selected_arrays = sel["arr_id"].unique()

        for id in selected_arrays:
            arr_sel = sel[sel["arr_id"] == id]
            color = np.random.rand(3)
            sources_ax.scatter(
                arr_sel["x"],
                arr_sel["y"],
                arr_sel["z"],
                c=color,
                alpha=0.8,
                s=50,
                label=f"array {id} - {arr_sel['shape'].iloc[0]} [{arr_sel['n_mic'].iloc[0]} mic / {arr_sel['sep'].iloc[0]} sep / {arr_sel['depth'].iloc[0]} depth]",
            )
            for _, row in arr_sel.iterrows():
                sources_ax.text(
                    row["x"], row["y"], row["z"], (str(id) + "." + str(row["mic_id"]))
                )
                sources_ax.plot(
                    [0, row["x"]],
                    [0, row["y"]],
                    [0, row["z"]],
                    c=color,
                    alpha=0.5,
                )
    sources_ax.set_title(f"{type} localized sources")
    sources_ax.legend()
    sources_ax.set_xlabel("x")
    sources_ax.set_ylabel("y")
    sources_ax.set_zlabel("z")
    sources_ax.set_box_aspect((2, 2, 1))

    plt.show()


# %% generate arrays
MIC_SEPARATION = [0.2, 1.5]  # in meters
MIC_DEPTH = [0.5, 1]  # in meters
N_MIC = [4]
ARRAY_SHAPE = ["star"]  # "star", "prism", "paraboloid", "random"]

arrays = []

arr_id = 0

for shape in ARRAY_SHAPE:
    for n_mic in N_MIC:
        for sep in MIC_SEPARATION:
            for depth in MIC_DEPTH:
                array = get_array(shape, n_mic, sep, depth)
                for mic_id, (x, y, z) in enumerate(array):
                    arrays.append((arr_id, shape, n_mic, sep, depth, mic_id, x, y, z))
                arr_id += 1

# create dataframe
array_df = pd.DataFrame(
    arrays,
    columns=["arr_id", "shape", "n_mic", "sep", "depth", "mic_id", "x", "y", "z"],
)

# %% create sources volume

SOURCES_DENSITY = 1  # distances on surface and radius
MAX_DISTANCE = 10.0  # in meters
MIN_DISTANCE = 2.0  # in meters
sources = []

r_values = np.linspace(
    MIN_DISTANCE, MAX_DISTANCE, int((MAX_DISTANCE - MIN_DISTANCE) * SOURCES_DENSITY + 1)
)
for r in r_values:

    theta_values = np.linspace(0, np.pi / 2, int(r * SOURCES_DENSITY) + 1)
    for theta in theta_values:
        phi_values = np.linspace(
            0,
            2 * np.pi,
            max((int(r * np.sin(theta) * 4 * SOURCES_DENSITY), 1)),
            endpoint=False,
        )
        for phi in phi_values:
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            sources.append((x, y, z, r, theta, phi))

sources_df = pd.DataFrame(sources, columns=["x", "y", "z", "r", "theta", "phi"])


sources_fig = plt.figure()
sources_ax = sources_fig.add_subplot(111, projection="3d")
sources_ax.scatter(0, 0, 0, color="red", s=20)
src_radius = sources_df["r"].unique()
for r in src_radius:
    sel = sources_df[sources_df["r"] == r]
    sources_ax.scatter(sel["x"], sel["y"], sel["z"], s=5, label=f"r={r:.2f} m")
sources_ax.set_title("sources")
sources_ax.set_xlabel("x")
sources_ax.set_ylabel("y")
sources_ax.set_zlabel("z")
sources_ax.set_box_aspect((2, 2, 1))

plt.show()


# %% straight plot array
sel = array_df[array_df["arr_id"] < 3]  # select here arrays to be evaluated
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
selected_arrays = sel["arr_id"].unique()

for id in selected_arrays:
    arr_sel = sel[sel["arr_id"] == id]
    color = np.random.rand(3)
    ax.scatter(
        arr_sel["x"],
        arr_sel["y"],
        arr_sel["z"],
        c=color,
        alpha=0.8,
        s=50,
        label=f"array {id} - {arr_sel['shape'].iloc[0]} [{arr_sel['n_mic'].iloc[0]} mic / {arr_sel['sep'].iloc[0]} sep / {arr_sel['depth'].iloc[0]} depth]",
    )
    for _, row in arr_sel.iterrows():
        ax.text(row["x"], row["y"], row["z"], (str(id) + "." + str(row["mic_id"])))
        ax.plot(
            [0, row["x"]],
            [0, row["y"]],
            [0, row["z"]],
            c=color,
            alpha=0.5,
        )
ax.set_title(f"array shapes {selected_arrays}")
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_box_aspect((2, 2, 2))

plt.show()

# %% compute delays for each couple source microphone in any
sources_df["arr_id"] = np.nan
sources_df["pl_x"] = np.nan
sources_df["pl_y"] = np.nan
sources_df["pl_z"] = np.nan
sources_df["nl_x"] = np.nan
sources_df["nl_y"] = np.nan
sources_df["nl_z"] = np.nan
sources_df["pure_error"] = np.nan
sources_df["noisy_error"] = np.nan

sel = array_df[array_df["arr_id"] < 20]  # select here arrays to be evaluated
selected_arrays = sel["arr_id"].unique()
results_list = []

NOISE_AMOUNT = 0.001  # in meters
N_ITERATIONS = 10  # number of iterations for each source

for id in selected_arrays:
    c_sources_df = sources_df.copy()
    arr_sel = sel[sel["arr_id"] == id]
    mic_array = arr_sel[["x", "y", "z"]].to_numpy(dtype=float)
    for s_idx, src in c_sources_df.iterrows():
        source_pos = src[["x", "y", "z"]].to_numpy(dtype=float)
        computer_pure_errors = []
        computer_noisy_errors = []
        for _ in range(N_ITERATIONS):
            TOAs = []
            noisy_TOAs = []
            delays = []
            noisy_delays = []
            for mic in mic_array:
                distance = np.linalg.norm(source_pos - mic)
                toa = distance  # / 343.0
                TOAs.append(toa)
                noisy_TOAs.append(
                    toa + np.random.normal(0, NOISE_AMOUNT)
                )  # add noise to the delay
            delays = np.array(TOAs[1:] - TOAs[0])  # relative to mic 0
            noisy_delays = np.array(noisy_TOAs[1:] - noisy_TOAs[0])  # relative to mic 0
            pure_localization = np.array(
                mpr.tristar_mellen_pachter(mic_array, delays), dtype=float
            ).reshape(-1)
            noisy_localization = np.array(
                mpr.tristar_mellen_pachter(mic_array, noisy_delays), dtype=float
            ).reshape(-1)

            pure_error = (
                np.linalg.norm(pure_localization[:3] - source_pos)
                if pure_localization.size > 0
                else np.nan
            )
            noisy_error = (
                np.linalg.norm(noisy_localization[:3] - source_pos)
                if noisy_localization.size > 0
                else np.nan
            )
            computer_pure_errors.append(pure_error)
            computer_noisy_errors.append(noisy_error)
        # add results to sources DF
        c_sources_df.loc[s_idx, "arr_id"] = int(id)

        c_sources_df.loc[s_idx, "pl_x"] = (
            pure_localization[0] if pure_localization.size > 0 else np.nan
        )
        c_sources_df.loc[s_idx, "pl_y"] = (
            pure_localization[1] if pure_localization.size > 1 else np.nan
        )
        c_sources_df.loc[s_idx, "pl_z"] = (
            pure_localization[2] if pure_localization.size > 2 else np.nan
        )
        c_sources_df.loc[s_idx, "nl_x"] = (
            noisy_localization[0] if noisy_localization.size > 0 else np.nan
        )
        c_sources_df.loc[s_idx, "nl_y"] = (
            noisy_localization[1] if noisy_localization.size > 1 else np.nan
        )
        c_sources_df.loc[s_idx, "nl_z"] = (
            noisy_localization[2] if noisy_localization.size > 2 else np.nan
        )
        c_sources_df.loc[s_idx, "pure_error"] = np.mean(computer_pure_errors)
        c_sources_df.loc[s_idx, "noisy_error"] = np.mean(computer_noisy_errors)

    results_list.append(c_sources_df)

sources_df = pd.concat(results_list, ignore_index=True)

# %% compute statistics
from scipy.interpolate import make_interp_spline, PchipInterpolator
from pathlib import Path
import matplotlib.colors as mcolors

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sim_id = 1
while (RESULTS_DIR / f"SIM_{sim_id}").exists():
    sim_id += 1
SIM_DIR = RESULTS_DIR / f"SIM_{sim_id}"
SIM_DIR.mkdir(parents=True, exist_ok=True)


FALSE_LOC_TOLERANCE = 0.8  # percent of distance, discard localization errors above this threshold as false localizations
theta_intervals = np.linspace(0, np.pi / 2, 5)
r_labels = [f"_r_{r:.2f}" for r in r_values]
theta_labels = [
    f"_theta_{theta_intervals[t]*180/np.pi:.0f}_{theta_intervals[t+1]*180/np.pi:.0f}"
    for t in range(len(theta_intervals) - 1)
]
results = pd.DataFrame(
    columns=[
        "arr_id",
        "pure_error_mean",
        "pure_error_median",
        "noisy_error_mean",
        "noisy_error_median",
        "pure_false_loc_count",
        "noisy_false_loc_count",
    ]
)

for lab in r_labels:
    results[f"pure{lab}"] = np.nan
    results[f"noisy{lab}"] = np.nan

for lab in theta_labels:
    results[f"pure{lab}"] = np.nan
    results[f"noisy{lab}"] = np.nan
colmap = mcolors.LinearSegmentedColormap.from_list("green_red", ["green", "red"])
# colmap= plt.cm.PiYG
col_norm = mcolors.Normalize(vmin=0, vmax=100)

for array_id in sources_df["arr_id"].unique():
    arr_sel = sources_df[sources_df["arr_id"] == array_id]
    non_false_p = arr_sel[arr_sel["pure_error"] < FALSE_LOC_TOLERANCE * arr_sel["r"]]
    non_false_n = arr_sel[arr_sel["noisy_error"] < FALSE_LOC_TOLERANCE * arr_sel["r"]]
    pure_error_mean = non_false_p["pure_error"].mean()
    pure_error_median = arr_sel["pure_error"].median()
    noisy_error_mean = non_false_n["noisy_error"].mean()
    noisy_error_median = arr_sel["noisy_error"].median()
    pure_false_loc_count = arr_sel.shape[0] - non_false_p.shape[0]
    noisy_false_loc_count = arr_sel.shape[0] - non_false_n.shape[0]
    row_index = len(results)
    print(row_index, array_id)
    results.loc[row_index] = {
        "arr_id": array_id,
        "pure_error_mean": pure_error_mean,
        "pure_error_median": pure_error_median,
        "noisy_error_mean": noisy_error_mean,
        "noisy_error_median": noisy_error_median,
        "pure_false_loc_count": pure_false_loc_count,
        "noisy_false_loc_count": noisy_false_loc_count,
    }

    for i, r in enumerate(r_values):
        non_false_p_r = non_false_p[non_false_p["r"] == r]
        non_false_n_r = non_false_n[non_false_n["r"] == r]
        pure_error_mean_r = non_false_p_r["pure_error"].mean()
        noisy_error_mean_r = non_false_n_r["noisy_error"].mean()
        results.loc[row_index, f"pure{r_labels[i]}"] = pure_error_mean_r
        results.loc[row_index, f"noisy{r_labels[i]}"] = noisy_error_mean_r

    for i, t in enumerate(theta_intervals[:-1]):
        non_false_p_t = non_false_p[
            (non_false_p["theta"] >= t)
            & (non_false_p["theta"] < theta_intervals[i + 1])
        ]
        non_false_n_t = non_false_n[
            (non_false_n["theta"] >= t)
            & (non_false_n["theta"] < theta_intervals[i + 1])
        ]
        pure_error_mean_t = non_false_p_t["pure_error"].mean()
        noisy_error_mean_t = non_false_n_t["noisy_error"].mean()
        results.loc[row_index, f"pure{theta_labels[i]}"] = pure_error_mean_t
        results.loc[row_index, f"noisy{theta_labels[i]}"] = noisy_error_mean_t

    ####PLOT RESULTS###########

    img = plt.figure(figsize=(22, 16))
    gs = img.add_gridspec(
        3, 5, width_ratios=[1, 0.1, 1, 0.1, 1], height_ratios=[1, 0.1, 1]
    )

    ## plot array
    arr_ax = img.add_subplot(gs[0, 0], projection="3d")
    this_array = array_df[array_df["arr_id"] == array_id]
    color = np.random.rand(3)
    arr_ax.scatter(
        this_array["x"],
        this_array["y"],
        this_array["z"],
        c=color,
        alpha=0.8,
        s=50,
    )
    for _, row in this_array.iterrows():
        arr_ax.text(
            row["x"], row["y"], row["z"], (str(array_id) + "." + str(row["mic_id"]))
        )
        arr_ax.plot(
            [0, row["x"]],
            [0, row["y"]],
            [0, row["z"]],
            c=color,
            alpha=0.8,
        )
    arr_ax.set_title("Array shape")
    arr_ax.set_xlabel("x")
    arr_ax.set_ylabel("y")
    arr_ax.set_zlabel("z")
    arr_ax.set_box_aspect((2, 2, 2))

    # plot noisy localization
    noisy_ax = img.add_subplot(gs[0, 2], projection="3d")
    noisy_ax.set_title("Noisy Localization")
    noisy_ax.scatter(0, 0, 0, color="green", s=20)
    src = noisy_ax.scatter(
        non_false_n["x"],
        non_false_n["y"],
        non_false_n["z"],
        s=5,
        alpha=0.5,
        c=non_false_n["noisy_error"],
        cmap="viridis",
    )
    false_n = arr_sel.drop(non_false_n.index)
    false = noisy_ax.scatter(
        false_n["x"],
        false_n["y"],
        false_n["z"],
        s=5,
        alpha=0.5,
        c="red",
        label="false localizations",
    )
    fig.colorbar(src, ax=noisy_ax, pad=0.1, label="Noisy Localization Error (m)")
    noisy_ax.legend()
    noisy_ax.set_xlabel("x")
    noisy_ax.set_ylabel("y")
    noisy_ax.set_zlabel("z")
    noisy_ax.set_box_aspect((2, 2, 1))

    # plot error to distance
    PLOT_Y_LIMIT = 3
    error_ax = img.add_subplot(gs[2, 2])
    error_ax.set_title("Noisy Localization Error vs Distance")
    min_4 = arr_sel[arr_sel["noisy_error"] < PLOT_Y_LIMIT]
    error_ax.plot(min_4["r"], min_4["noisy_error"], "o", alpha=0.5)
    for r in r_values:
        more_4 = (
            1 - min_4[min_4["r"] == r].shape[0] / arr_sel[arr_sel["r"] == r].shape[0]
        ) * 100
        color = colmap(col_norm(more_4))
        error_ax.text(
            r - 0.1, PLOT_Y_LIMIT - 0.1, f"{more_4:.0f}%", fontsize=10, color=color
        )
    error_ax.set_ylim(0, PLOT_Y_LIMIT)
    error_ax.set_xlabel("Distance (m)")
    error_ax.set_ylabel("Noisy Localization Error (m)")

    # polar plot of error vs angle
    polar_ax = img.add_subplot(gs[0, 4], projection="polar")
    polar_ax.set_title("Noisy Localization Error vs Theta")
    for r in r_values:
        non_false_n_r = non_false_n[non_false_n["r"] == r]
        non_false_n_r = non_false_n_r.sort_values(by="theta")
        theta = []
        error = []
        for t in non_false_n_r["theta"].unique():
            non_false_n_r_t = non_false_n_r[non_false_n_r["theta"] == t]
            theta.append(t)
            error.append(non_false_n_r_t["noisy_error"].mean())
        theta = np.array(theta)
        error = np.maximum(np.array(error), 0.001)
        if len(theta) > 2:
            theta_smooth = np.linspace(theta.min(), theta.max(), 200)
            # spl = make_interp_spline(theta, error, k=2)
            spl = PchipInterpolator(theta, error)
            error_smooth = spl(theta_smooth)
            polar_ax.plot(
                theta_smooth,
                error_smooth,
                label=f"r={r:.2f} m",
                alpha=0.5,
            )
    polar_ax.set_thetalim(0, np.pi / 2)
    polar_ax.set_rscale("log")
    polar_ax.set_rmin(0.001)
    polar_ax.set_rmax(10)
    polar_ax.set_theta_zero_location("N")
    polar_ax.set_theta_direction(-1)
    polar_ax.set_xlabel("Theta (rad)")
    polar_ax.set_ylabel("Noisy Localization Error (m)")
    polar_ax.legend()

    # polar plot along phi and distance
    phi_ax = img.add_subplot(gs[2, 4], projection="polar")
    phi_ax.set_title("Noisy Localization Error vs Phi [log scale]")
    sorted_z = arr_sel.sort_values(by="z")
    z_span = len(sorted_z) // 3
    for i in range(3):
        z_sel = sorted_z.iloc[i * z_span : (i + 1) * z_span]
        z_min = z_sel["z"].min()
        z_max = z_sel["z"].max()
        phi_intervals = np.linspace(0, 2 * np.pi, 41)
        phi_delta = phi_intervals[1] - phi_intervals[0]
        error_phi = []
        phi_centers = []
        for j in range(len(phi_intervals) - 1):
            phi_min = phi_intervals[j]
            phi_max = phi_intervals[j + 1]

            subset = z_sel[
                (z_sel["phi"] >= phi_min)
                & (z_sel["phi"] < phi_max)
                & (z_sel["noisy_error"] != np.nan)
            ]
            if not subset.empty:
                phi_err_value = abs(subset["noisy_error"]).median()
                error_phi.append(phi_err_value)
                phi_centers.append((phi_min + phi_max) / 2)
        # pop NaN values from error_phi and phi_centers, spline will smooth out the missimg values
        for i in range(len(error_phi)):
            if np.isnan(error_phi[i]):
                error_phi.pop(i)
                phi_centers.pop(i)

        if len(phi_centers) > 2:
            phi_centers = np.array(phi_centers)
            error_phi = np.array(error_phi)
            phi_smooth = np.linspace(phi_centers.min(), phi_centers.max(), 200)
            # spl = make_interp_spline(phi_centers, np.log10(error_phi), k=2)
            spl = PchipInterpolator(phi_centers, error_phi)
            error_smooth = spl(phi_smooth)
            phi_ax.plot(
                phi_smooth,
                error_smooth,
                label=f"z={z_min:.2f}-{z_max:.2f} m",
                alpha=0.5,
            )

    for _, row in this_array.iterrows():
        line_angle = np.arctan2(row["y"], row["x"])
        if line_angle < 0:
            line_angle += 2 * np.pi
        phi_ax.axvline(
            line_angle,
            # np.sqrt(row["x"] ** 2 + row["y"] ** 2) * np.cos(row["z"]),
            linestyle="--",
            linewidth=1,
            color="red",
        )
    phi_ax.set_thetalim(0, 2 * np.pi)

    phi_ax.set_rscale("log")
    phi_ax.set_rmin(0.001)
    phi_ax.set_rmax(10)
    phi_ax.set_theta_zero_location("N")
    phi_ax.set_theta_direction(-1)
    phi_ax.set_xlabel("Phi (rad)")
    phi_ax.legend()

    # image settings

    img.suptitle(
        f"Array {int(array_id)} - shape: {this_array['shape'].iloc[0]} [{this_array['n_mic'].iloc[0]} mic / {this_array['sep'].iloc[0]} sep / {this_array['depth'].iloc[0]} depth], Mean error: {noisy_error_mean:.3f} m, Dropped: {noisy_false_loc_count}",
        fontsize=18,
        fontweight="bold",
    )

    img.savefig(
        SIM_DIR
        / f"array_{int(array_id)}_{this_array['shape'].iloc[0]}_{this_array['n_mic'].iloc[0]}mic_{this_array['sep'].iloc[0]}sep_{this_array['depth'].iloc[0]}dep.png",
        dpi=300,
    )
    img.tight_layout()
    plt.show()

compare = plt.figure(figsize=(10, 5))
compare.suptitle("Comparison of evaluated arrays", fontsize=18, fontweight="bold")
gg = compare.add_gridspec(1, 2, width_ratios=[1, 1])
me = compare.add_subplot(gg[0, 0])
me.set_title("Median Localization Error across Arrays")
me.bar(results["arr_id"], results["noisy_error_median"], alpha=0.5)
me.xaxis.set_ticks(results["arr_id"].unique())

fl = compare.add_subplot(gg[0, 1])
fl.set_title(
    f"False Localization Count across Arrays (out of {c_sources_df.shape[0]} sources)"
)
fl.bar(results["arr_id"], results["noisy_false_loc_count"], alpha=0.5)
fl.xaxis.set_ticks(results["arr_id"].unique())

compare.savefig(SIM_DIR / f"comparison_arrays.png", dpi=300)
compare.tight_layout()
plt.show()

results.to_csv(SIM_DIR / f"results.csv", index=False)
array_df.to_csv(SIM_DIR / f"array_shapes.csv", index=False)
# %%
