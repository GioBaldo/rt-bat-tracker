"""
Plotting utilities for localisation analytics.

Figures are built with the object-oriented matplotlib API (matplotlib.figure.Figure)
instead of pyplot, so they don't touch the GUI backend and can be created
safely from any thread. Use fig.savefig(...) to save them; no plt.close() needed.
To display one interactively pass a pyplot figure:
    fig = plt.figure(); plot_cross_correlations(analytics, chunk, fig=fig); plt.show()
"""

import numpy as np
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.figure import Figure


def plot_cross_correlations(
    analytics, chunk: np.ndarray = None, fig: Figure = None
) -> Figure:
    """Plot of the analytics metrics and of the separate audio channels.

    Layout (5 columns):
    - Row 1: 5 separate subplots (one per audio channel), only if chunk is given.
    - Rows 2+: cross-correlation subplots arranged on 5 columns, with border
      colored by zero-sum error (green = 0, red = max).

    Args:
        analytics: LocAnalytics object (cross_correlations, channel_pairs, TDOA, cc_errors)
        chunk (np.ndarray, optional): audio chunk (n_samples, n_channels)
        fig (Figure, optional): existing figure to draw into, e.g. plt.figure()
            when you want to display it with plt.show(). If None a new
            pyplot-free Figure is created (safe for saving from any thread).

    Returns:
        matplotlib.figure.Figure
    """
    num_pairs = len(analytics.cross_correlations)
    cols = 5

    cc_rows = (num_pairs + cols - 1) // cols if num_pairs > 0 else 0
    total_rows = 1 + cc_rows  # 1 audio row + CC rows

    if fig is None:
        fig = Figure()
    fig.set_size_inches(18, 2.5 * total_rows)
    fig.set_layout_engine("constrained")
    fig.suptitle(
        "Localisation Analytics - Audio Channels & Cross-Correlations",
        fontsize=15,
        fontweight="bold",
    )

    gs = fig.add_gridspec(total_rows, cols)

    # ----------------------------------------------------
    # ROW 1: audio channels
    # ----------------------------------------------------
    if chunk is not None:
        num_samples, num_channels = chunk.shape
        time_axis = np.arange(num_samples)
        colors_audio = matplotlib.colormaps["Set1"](
            np.linspace(0, 1, max(num_channels, 5))
        )

        # mic_errors is indexed by position in significant_channels, not by channel number
        significant = [int(c) for c in getattr(analytics, "significant_channels", [])]
        mic_errors = list(getattr(analytics, "mic_errors", []))
        ch_error = {
            ch: mic_errors[i] for i, ch in enumerate(significant) if i < len(mic_errors)
        }
        selected = {int(c) for c in getattr(analytics, "selected_mics", [])}

        max_mic_err = max(ch_error.values()) if ch_error else 0.0
        mic_norm = mcolors.Normalize(vmin=0.0, vmax=max_mic_err if max_mic_err > 0 else 1.0)
        mic_cmap = matplotlib.colormaps["RdYlGn_r"]

        for ch in range(min(num_channels, cols)):
            ax_ch = fig.add_subplot(gs[0, ch])
            in_use = ch in selected
            ax_ch.plot(
                time_axis,
                chunk[:, ch],
                color=colors_audio[ch],
                linewidth=0.8,
                alpha=1.0 if in_use else 0.35,
            )
            ax_ch.set_title(
                f"Channel {ch}", fontsize=10, fontweight="semibold", color="navy"
            )
            ax_ch.grid(True, linestyle=":", alpha=0.6)
            ax_ch.tick_params(axis="both", labelsize=8)

            # status label: IN_USE / DISCARDED + zero-sum mic error (µs)
            err = ch_error.get(ch)
            status = "IN_USE" if in_use else "DISCARDED"
            err_txt = f"{err * 1e6:.1f} µs" if err is not None else "n/a"
            box_color = mic_cmap(mic_norm(err)) if err is not None else "grey"

            for spine in ax_ch.spines.values():
                spine.set_edgecolor(box_color)
                spine.set_linewidth(2.5)

            ax_ch.text(
                0.03,
                0.72,
                f"{status}\nErr: {err_txt}",
                transform=ax_ch.transAxes,
                fontsize=7,
                fontweight="bold" if in_use else "normal",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white" if in_use else "lightgrey",
                    alpha=0.85,
                    edgecolor=box_color,
                    lw=1.2,
                ),
            )
            if ch == 0:
                ax_ch.set_ylabel("Amplitude", fontsize=8)
            ax_ch.set_xlabel("Samples", fontsize=8)

    # ----------------------------------------------------
    # ROWS 2+: cross-correlations grid
    # ----------------------------------------------------
    if num_pairs > 0:
        cc_colors = matplotlib.colormaps["viridis"](np.linspace(0.1, 0.9, num_pairs))

        # error scale normalisation: green = 0, red = max
        max_err = (
            np.max(analytics.cc_errors)
            if len(analytics.cc_errors) > 0 and np.max(analytics.cc_errors) > 0
            else 1.0
        )
        norm = mcolors.Normalize(vmin=0.0, vmax=max_err)
        cmap_error = matplotlib.colormaps["RdYlGn_r"]

        for i, cc in enumerate(analytics.cross_correlations):
            r = 1 + (i // cols)
            c = i % cols

            ax_cc = fig.add_subplot(gs[r, c])
            pair = analytics.channel_pairs[i] if i < len(analytics.channel_pairs) else (i,)

            current_err = analytics.cc_errors[i] if i < len(analytics.cc_errors) else 0.0
            border_color = cmap_error(norm(current_err))

            for spine in ax_cc.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2.5)

            ax_cc.plot(cc, color=cc_colors[i], linewidth=1.1)

            # marker on the max peak (estimated lag)
            max_idx = np.argmax(cc)
            ax_cc.plot(max_idx, cc[max_idx], "ro", markersize=3.5)
            ax_cc.axvline(x=max_idx, color="red", linestyle="--", alpha=0.5, linewidth=0.8)

            ax_cc.set_title(f"Pair {pair}", fontsize=9, fontweight="semibold")
            ax_cc.grid(True, linestyle=":", alpha=0.6)
            ax_cc.tick_params(axis="both", labelsize=8)

            if i < len(analytics.TDOA):
                tdoa_us = analytics.TDOA[i] * 1e6
                ax_cc.text(
                    0.03,
                    0.72,
                    f"TDOA: {tdoa_us:.1f} µs\nErr: {current_err * 1e6:.1f} µs",
                    transform=ax_cc.transAxes,
                    fontsize=7,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        alpha=0.85,
                        edgecolor=border_color,
                        lw=1.2,
                    ),
                )

            if c == 0:
                ax_cc.set_ylabel("Correlation", fontsize=8)
            if r == total_rows - 1:
                ax_cc.set_xlabel("Lag", fontsize=8)

    return fig
