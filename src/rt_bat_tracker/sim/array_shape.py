# %% imports
# %matplotlib qt #solo se giro in locale
# %%
import sim_localisation_mpr2003 as mpr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Simulate array shapes and computes the localization error with respect to a TDOA error
    This relates the array shape to its sensitivity to fainth errors on the TDOA which can be due to incorrect positioning of microphones or inaccuracy in computing delays 

    Reference coordinate system is (0, 0, 0) at the center of the array, with the z-axis pointing outwards. The array is assumed to be in the xy-plane at z=0.
"""
# %% defaults
MIC_SEPARATION = [0.5, 1.0]  # in meters
MIC_DEPTH = [1.0]  # in meters
N_MIC = [6, 7]
ARRAY_SHAPE = ["star"]  # , "prism", "paraboloid", "random"]


arrays = []
results = []


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

    pass


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


def plot_sources(sources_df, type="src"):
    sources_fig = plt.figure()
    sources_ax = sources_fig.add_subplot(111, projection="3d")
    sources_ax.scatter(0, 0, 0, color="red", s=20)
    src_radius = sources_df["r"].unique()
    match type:
        case "src":
            for r in src_radius:
                eq_r_df = sources_df[sources_df["r"] == r]
                sources_ax.scatter(
                    eq_r_df["pl_x"],
                    eq_r_df["pl_y"],
                    eq_r_df["pl_z"],
                    s=5,
                    label=f"r={r:.2f} m",
                )
        case "pure":
            for r in src_radius:
                eq_r_df = sources_df[sources_df["r"] == r]
                sources_ax.scatter(
                    eq_r_df["pl_x"],
                    eq_r_df["pl_y"],
                    eq_r_df["pl_z"],
                    s=5,
                    label=f"r={r:.2f} m",
                )
        case "noisy":
            for r in src_radius:
                eq_r_df = sources_df[sources_df["r"] == r]
                sources_ax.scatter(
                    eq_r_df["nl_x"],
                    eq_r_df["nl_y"],
                    eq_r_df["nl_z"],
                    s=5,
                    label=f"r={r:.2f} m",
                )
        case "pure_c":
            sources_ax.scatter(
                sources_df["x"],
                sources_df["y"],
                sources_df["z"],
                s=5,
                color=sources_df["pure_error"],
                cmap="viridis",
            )
        case "noisy_c":
            sources_ax.scatter(
                sources_df["x"],
                sources_df["y"],
                sources_df["z"],
                s=5,
                color=sources_df["noisy_error"],
                cmap="viridis",
            )
    sources_ax.set_title(f"{type} localized sources")
    sources_ax.set_xlabel("x")
    sources_ax.set_ylabel("y")
    sources_ax.set_zlabel("z")
    sources_ax.set_box_aspect((2, 2, 1))

    plt.show()


# %% generate arrays
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
MIN_DISTANCE = 1.0  # in meters
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

# %% straight plot sources

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
sel = array_df[array_df["n_mic"] == 7]  # select here arrays to be evaluated
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

# %% compute delays for each couple source microphone in any array
sources_df["pl_x"] = np.nan
sources_df["pl_y"] = np.nan
sources_df["pl_z"] = np.nan
sources_df["nl_x"] = np.nan
sources_df["nl_y"] = np.nan
sources_df["nl_z"] = np.nan
sources_df["pure_error"] = np.nan
sources_df["noisy_error"] = np.nan

sel = array_df[array_df["arr_id"] == 0]  # select here arrays to be evaluated
selected_arrays = sel["arr_id"].unique()

NOISE_AMOUNT = 0.0001  # in seconds
for id in selected_arrays:
    arr_sel = sel[sel["arr_id"] == id]
    mic_array = arr_sel[["x", "y", "z"]].to_numpy(dtype=float)
    for s_idx, src in sources_df.iterrows():
        source_pos = src[["x", "y", "z"]].to_numpy(dtype=float)
        delays = []
        noisy_delays = []
        for mic in mic_array:
            distance = np.linalg.norm(source_pos - mic)
            delay = distance / 343.0
            delays.append(delay)
            noisy_delays.append(
                delay + np.random.normal(0, NOISE_AMOUNT)
            )  # add noise to the delay
        delays = np.array(delays[1:] - delays[0])  # relative to mic 0
        noisy_delays = np.array(noisy_delays[1:] - noisy_delays[0])  # relative to mic 0
        pure_localization = mpr.mellen_pachter_raquet_2003(mic_array, delays)
        noisy_localization = mpr.mellen_pachter_raquet_2003(mic_array, noisy_delays)
        pure_error = (
            np.linalg.norm(pure_localization - source_pos)
            if pure_localization.size > 0
            else np.nan
        )
        noisy_error = (
            np.linalg.norm(noisy_localization - source_pos)
            if noisy_localization.size > 0
            else np.nan
        )
        sources_df.loc[s_idx, "pl_x"] = (
            pure_localization[0] if pure_localization.size > 0 else np.nan
        )
        sources_df.loc[s_idx, "pl_y"] = (
            pure_localization[1] if pure_localization.size > 1 else np.nan
        )
        sources_df.loc[s_idx, "pl_z"] = (
            pure_localization[2] if pure_localization.size > 2 else np.nan
        )
        sources_df.loc[s_idx, "nl_x"] = (
            noisy_localization[0] if noisy_localization.size > 0 else np.nan
        )
        sources_df.loc[s_idx, "nl_y"] = (
            noisy_localization[1] if noisy_localization.size > 1 else np.nan
        )
        sources_df.loc[s_idx, "nl_z"] = (
            noisy_localization[2] if noisy_localization.size > 2 else np.nan
        )
        sources_df.loc[s_idx, "pure_error"] = pure_error
        sources_df.loc[s_idx, "noisy_error"] = noisy_error

# %% plot purely localized sources

sources_fig = plt.figure()
sources_ax = sources_fig.add_subplot(111, projection="3d")
sources_ax.scatter(0, 0, 0, color="red", s=20)
src_radius = sources_df["r"].unique()
for r in src_radius:
    eq_r_df = sources_df[sources_df["r"] == r]
    sources_ax.scatter(
        eq_r_df["pl_x"], eq_r_df["pl_y"], eq_r_df["pl_z"], s=5, label=f"r={r:.2f} m"
    )
sources_ax.set_title("pure localized sources")
sources_ax.set_xlabel("x")
sources_ax.set_ylabel("y")
sources_ax.set_zlabel("z")
sources_ax.set_box_aspect((2, 2, 1))

plt.show()

# %% plot noisy localized sources

sources_fig = plt.figure()
sources_ax = sources_fig.add_subplot(111, projection="3d")
sources_ax.scatter(0, 0, 0, color="red", s=20)
src_radius = sources_df["r"].unique()
for r in src_radius:
    eq_r_df = sources_df[sources_df["r"] == r]
    sources_ax.scatter(
        eq_r_df["nl_x"], eq_r_df["nl_y"], eq_r_df["nl_z"], s=5, label=f"r={r:.2f} m"
    )
sources_ax.set_title("noisy localized sources")
sources_ax.set_xlabel("x")
sources_ax.set_ylabel("y")
sources_ax.set_zlabel("z")
sources_ax.set_box_aspect((2, 2, 1))

plt.show()

# %%
