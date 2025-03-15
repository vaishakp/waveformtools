""" Compare the waveforms and modes """

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Plot settings
fontsize = 16
labelsize = 16
labelpad = 14

matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# plt.rcParams.update({'font.size': fontsize})
plt.rcParams.update({"figure.figsize": (8, 6)})
# plt.rcParams.update({"axes.grid" : True})
plt.rcParams.update({"axes.labelpad": labelpad})
plt.rcParams.update({"axes.labelsize": labelsize})
plt.rcParams.update({"figure.autolayout": True})
plt.rcParams.update({"grid.alpha": 0.3})
plt.rcParams.update({"grid.alpha": 0.3})
# plt.rcParams.update({"font.weight" : "bold"})
# plt.rcParams.update({"axes.labelweight" : "bold"})
plt.rcParams.update({"font.style": "normal"})
plt.rcParams.update({"legend.markerscale": 9})

# from waveformtools.waveforms import modes_array


def plot_modes(
    wf1,
    nmodes=3,
    save_fig=False,
    xlim=[-1200, 100],
    ylim="auto",
    nstop=1,
    plot22=False,
    figsize=(18, 12)
):
    """Plot the first `nmodes` dominant modes of
    the input waveforms

    Parameters
    ----------
    wf : modes_array
         The list of `modes_array` waveforms
         to plot.
    nmodes : int
             The number of modes to plot.
    xlim : list
           [xmin, xmax] limits to plot.
    tol : float
          The tolerance to detect the modes.

    Returns
    -------
    Plots
    """

    from waveformtools.waveformtools import xtract_cphase
    # Start from l=2.
    modes_to_plot = wf1.modes_list[:]
    # Order the modes as per 1st waveform modes_array object.
    modes_list = []
    mode_amps = []

    for item in modes_to_plot:
        ell, emm_values = item
        for emm in emm_values:

            mode_values = wf1.mode(ell, emm)
            mode_amp = np.absolute(mode_values)
            max_mode_value = np.amax(mode_amp[:-nstop])
            mode_phase = xtract_cphase(mode_values.real, mode_values.imag)
            mode_amps.append(max_mode_value)
            modes_list.append(f"l{ell}m{emm}")

    # Sort the mode amps
    args_sorted = np.argsort(np.array(mode_amps))[::-1]
    mode_list_sorted = [modes_list[index] for index in args_sorted]
    modes_to_plot = []

    for item in mode_list_sorted:
        c1, c2 = item.split('m')
        ell = int(c1[1:])
        emm = int(c2)

        if emm > 0:
            modes_to_plot.append([ell, [emm]])

    # Plots
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=figsize)
    # For amplitudes
    #fig1, ax1 = plt.subplots()
    ax[0].set_yscale("log")
    # For phases
    #fig2, ax2 = plt.subplots()
    #ax2.set_yscale("log")
    for item in modes_to_plot[:nmodes]:
        ell, emm_list = item
        for emm in emm_list:
            mode_values = wf1.mode(ell, emm)
            mode_amp = np.absolute(mode_values)
            max_mode_value = np.amax(mode_values)
            mode_phase = xtract_cphase(mode_values.real, mode_values.imag)
            ax[0].scatter(
                wf1.time_axis[:],
                mode_amp[:],
                label=rf"$\ell${ell}$m${emm}",
                s=1,
                alpha=0.2 * abs(emm) % 1,
            )
            ax[1].scatter(
                wf1.time_axis[:],
                abs(mode_phase[:]),
                label=rf"$\ell${ell}$m${emm}",
                s=1,
            )

    if plot22:
        ell = 2
        emm = 2
        mode_values = wf1.mode(ell, emm)
        mode_amp = np.absolute(mode_values)
        max_mode_value = np.amax(mode_values)

        mode_phase = xtract_cphase(mode_values.real, mode_values.imag)

        ax[0].scatter(
            wf1.time_axis[:],
            mode_amp[:],
            label=rf"$\ell${ell}$m${emm}",
            s=1,
            alpha=0.2 * abs(emm) % 1,
        )
        ax[1].scatter(
            wf1.time_axis[:],
            abs(mode_phase[:]),
            label=rf"$\ell${ell}$m${emm}",
            s=1,
        )

    ax[0].grid(which="both")
    ax[1].grid(which="both")
    ax[0].legend()
    #ax[1].legend()
    plt.tight_layout()
    ax[0].set_xlabel("t/M")
    #ax[1].set_ylabel(r"$\vert$" + wf1.label + r"$^{(\ell m)}\vert$")
    #ax[0].set_xlabel("t/M")
    ax[1].set_ylabel(r"$\Phi^{(\ell m)}$")
    #ax2.set_xlim(*xlim)
    ax[0].set_xlim(*xlim)
    # fig1.savefig('figures/waveform_extrapolation/amp_evol_modes_q1a0.pdf')
    # fig2.savefig('figures/waveform_extrapolation/phase_evol_modes_q1a0.pdf')
    if ylim != "auto":
        # ax2.set_ylim(
        #   *ylim
        # )
        ax[0].set_ylim(
            *ylim,
        )

    if save_fig:
        fig.savefig(f"{wf1.label}_waveform_modes_amp_phase.pdf")
        #fig2.savefig(f"{wf1.label}_waveform_phase_modes.pdf")

    #plt.show()
    return fig, ax

def plot_mode_differences(
    waveforms,
    nmodes=3,
    save_fig=False,
    xlabel="t/M",
    ylabel=r"r\Psi_{4}^{(\ell m)}",
    labels=None,
    xlim=[-1000, 100],
):
    """Plot the fractional difference of the first `nmodes`
    dominant modes of the input waveforms.

    Parameters
    ----------
    waveforms : modes_array
                The list of `modes_array` waveforms
                to plot.
    nmodes : int
             The number of modes to plot.
    tol : float
          The tolerance to detect the modes.

    Returns
    -------
    Plots.
    """

    # For phase computation.
    from waveformtools.waveformtools import xtract_cphase

    # List of Modes list to iterate over.
    modes_list_all = [wfx.modes_list for wfx in waveforms]
    modes_list_len = [len(item) for item in modes_list_all]

    # Get the modes list that has the smallest number
    # of modes.
    mloc = np.argmin(modes_list_len)

    modes_list = modes_list_all[mloc]

    # Prepare the labels to plot.
    if labels is None:
        labels = [str(item) for item in np.arange(len(waveforms))]

    # The defeult waveform to compare others with
    wf0 = waveforms[0]

    # Start from l=2.
    modes_to_plot = modes_list[:]

    # Sort the modes as per 1st waveform modes_array object mode strenghts.
    modes_list = []
    mode_amps = []

    for item in modes_to_plot:
        ell, emm_values = item

        for emm in emm_values:
            mode_values = wf0.mode(ell, emm)
            mode_amp = np.absolute(mode_values)
            max_mode_value = np.amax(mode_amp)

            # mode_phase = xtract_cphase(mode_values.real, mode_values.imag)

            mode_amps.append(max_mode_value)

            modes_list.append(f"l{ell}m{emm}")

    # Sort the mode amps

    args_sorted = np.argsort(np.array(mode_amps))[::-1]
    mode_list_sorted = [modes_list[index] for index in args_sorted]

    modes_to_plot = []

    # message(mode_list_sorted)
    for item in mode_list_sorted:
        ell = int(item[1])
        emm = int(item[3:])
        if emm > 0:
            modes_to_plot.append([ell, [emm]])

    # Prepare matplotlib env

    # For amplitudes
    fig1, ax1 = plt.subplots()
    ax1.set_yscale("log")

    # For phases
    fig2, ax2 = plt.subplots()
    ax2.set_yscale("log")

    # Find the mode with the smallest taxis extent.
    dlens = [wfx.data_len for wfx in waveforms]
    # Get the time axis limits
    tmins = [min(wfx.time_axis) for wfx in waveforms]
    tmaxs = [max(wfx.time_axis) for wfx in waveforms]

    # Find the smallest limits.
    tmin_loc = np.argmax(tmins)
    tmax_loc = np.argmin(tmaxs)

    tabs_min = tmins[tmin_loc]
    tabs_max = tmaxs[tmax_loc]

    # Construct new time axis with shortest available time limits.
    new_time_axis = np.linspace(tabs_min, tabs_max, np.min(dlens))
    new_delta_t = new_time_axis[1] - new_time_axis[0]

    from scipy.interpolate import interp1d

    start_time = 10
    dindex = 10
    # Interpolate
    cumulative_diff = {}
    for item in modes_to_plot[:nmodes]:
        ell, emm_list = item
        for emm in emm_list:
            mode_values0 = wf0.mode(ell, emm)
            mode_amp0 = np.absolute(mode_values0)

            mode_phase0 = xtract_cphase(mode_values0.real, mode_values0.imag)

            mode_amp_int_fun0 = interp1d(wf0.time_axis, mode_amp0)
            mode_amp_resam0 = mode_amp_int_fun0(new_time_axis)

            mode_phase_int_fun0 = interp1d(wf0.time_axis, mode_phase0)
            mode_phase_resam0 = mode_phase_int_fun0(new_time_axis)

            for index, wfx in enumerate(waveforms[1:]):
                label = wfx.label  # labels[index+1]

                mode_valuesx = wfx.mode(ell, emm)
                mode_ampx = np.absolute(mode_valuesx)
                mode_phasex = xtract_cphase(
                    mode_valuesx.real, mode_valuesx.imag
                )

                mode_amp_int_funx = interp1d(wfx.time_axis, mode_ampx)
                mode_amp_resamx = mode_amp_int_funx(new_time_axis)

                mode_phase_int_funx = interp1d(wfx.time_axis, mode_phasex)
                mode_phase_resamx = mode_phase_int_funx(new_time_axis)

                start_ind = int((-new_time_axis[0] - start_time) / new_delta_t)

                # message(start_ind)
                # Remove the man phase shift
                mean_phase_shift = np.mean(
                    mode_phase_resamx[start_ind : start_ind + dindex]
                    - mode_phase_resam0[start_ind : start_ind + dindex]
                )

                # mean_phase_shift = np.mean(
                #   mode_phase_resamx - mode_phase_resam0
                # )

                # message(mean_phase_shift)
                delta_mode_ampx = (
                    mode_amp_resamx - mode_amp_resam0
                ) / mode_amp_resam0
                delta_mode_phasex = (
                    mode_phase_resamx - mode_phase_resam0 - mean_phase_shift
                ) / mode_phase_resam0

                rms_amp_diff = np.sqrt(
                    np.sum(delta_mode_ampx**2) / (len(delta_mode_ampx))
                )
                rms_phase_diff = np.sqrt(
                    np.sum(delta_mode_phasex**2) / (len(delta_mode_phasex))
                )
                max_phase_diff = np.amax(np.absolute(delta_mode_phasex))
                max_phase_diff_loc = np.argmax(np.absolute(delta_mode_phasex))
                max_phase_diff_time = new_time_axis[max_phase_diff_loc]
                cumulative_diff.update(
                    {
                        f"l{ell}m{emm}": [
                            rms_amp_diff,
                            rms_phase_diff,
                            [max_phase_diff, max_phase_diff_time],
                        ]
                    }
                )
                max_mode_value = np.amax(mode_values)

            ax1.scatter(
                new_time_axis[:],
                abs(delta_mode_ampx[:]),
                label=rf"{label} $\ell${ell}$m${emm}",
                s=1,
                alpha=0.2 * abs(emm) % 1,
            )
            ax2.scatter(
                new_time_axis[:],
                abs(delta_mode_phasex[:]),
                label=rf"{label} $\ell${ell}$m${emm}",
                s=1,
            )

    ax1.grid(which="both")
    ax2.grid(which="both")
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(rf"$ \vert \delta {ylabel} \vert $")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(rf"$ \vert \delta \Phi ({ylabel}) \vert$")

    ax2.set_xlim(*xlim)
    ax1.set_xlim(*xlim)
    # fig1.savefig('figures/waveform_extrapolation/amp_evol_modes_q1a0.pdf')
    # fig2.savefig('figures/waveform_extrapolation/phase_evol_modes_q1a0.pdf')
    if save_fig:
        fig1.savefig("waveform_modes_amps_differences.pdf")
        fig2.savefig("waveform_modes_phases_differences.pdf")
    plt.show()

    return cumulative_diff
