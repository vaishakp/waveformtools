import numpy as np


def lal_fd_modes_to_td_modes(wfm_fd, undo_warp=True):
    """Convert LAL two-sided FD modes to TD modes.

    LAL FD modes are stored on an increasing two-sided frequency axis.  They
    are not in the shifted/conjugated/scaled convention expected by
    ``spectools.compute_ifft``.  The historical LAL loader stores FD data as
    ``conj(lal_data) / N``; this function first recovers the raw LAL samples
    and then applies the direct continuous inverse-sum normalization
    ``h(t) ~= df * sum_f H(f) exp(2 pi i f t)``.  The final conjugation matches
    waveformtools' TD mode convention, which also conjugates LAL TD modes at
    load time.
    """

    frequency_axis = np.asarray(wfm_fd.frequency_axis, dtype=float)
    if frequency_axis.ndim != 1 or frequency_axis.size != wfm_fd.data_len:
        raise ValueError("LAL FD modes require a 1D frequency axis.")
    if wfm_fd.data_len < 2:
        raise ValueError("Need at least two FD samples to transform to TD.")
    delta_f = float(wfm_fd.delta_f)
    if not np.isfinite(delta_f) or delta_f <= 0.0:
        raise ValueError("LAL FD modes require a positive finite delta_f.")

    from waveformtools.modes_array import ModesArray

    wfm_td = ModesArray(
        label=f"{wfm_fd.label} -> time_domain",
        ell_max=wfm_fd.ell_max,
        data_len=wfm_fd.data_len,
        modes_list=wfm_fd.modes_list,
        spin_weight=wfm_fd.spin_weight,
        time_axis=np.arange(wfm_fd.data_len, dtype=float)
        / (wfm_fd.data_len * delta_f),
    )
    wfm_td.create_modes_array()

    lal_fd_data = _recover_lal_fd_samples(wfm_fd._modes_data)
    time_data = np.conjugate(
        wfm_fd.data_len
        * delta_f
        * np.fft.ifft(np.fft.ifftshift(lal_fd_data, axes=-1), axis=-1)
    )
    wfm_td.set_mode_data(data=time_data)

    if undo_warp:
        wfm_td.undo_warp()
    return wfm_td


def fd_modes_to_td_modes(wfm_fd, undo_warp=True):
    """Inverse-FFT an FD ModesArray mode by mode."""
    from spectools.fourier.transforms import compute_ifft
    from waveformtools.modes_array import ModesArray

    wfm_td = ModesArray(
        label=f"{wfm_fd.label} -> time_domain",
        ell_max=wfm_fd.ell_max,
        data_len=wfm_fd.data_len,
        modes_list=wfm_fd.modes_list,
        spin_weight=wfm_fd.spin_weight,
    )
    wfm_td.create_modes_array()

    time_axis = None
    for ell, emm_list in wfm_fd.modes_list:
        for emm in emm_list:
            time_axis, time_data = compute_ifft(
                wfm_fd.mode(ell, emm),
                wfm_fd.delta_f,
            )
            wfm_td.set_mode_data(ell=ell, emm=emm, data=time_data)

    wfm_td._time_axis = time_axis
    if undo_warp:
        wfm_td.undo_warp()
    return wfm_td


def _recover_lal_fd_samples(stored_mode):
    """Undo waveformtools' historical LAL FD loader convention."""

    stored = np.asarray(stored_mode, dtype=np.complex128)
    return np.conjugate(stored) * stored.shape[-1]


def recenter_modes_at_peak(
    wfm_td,
    peak_target_frac=0.5,
    set_peak_time_to_zero=True,
):
    """Apply one common circular shift so the total-intensity peak is fixed."""
    if not (0.0 <= peak_target_frac <= 1.0):
        raise ValueError("peak_target_frac must be in [0, 1]")

    out = wfm_td.deepcopy()
    data_len = len(out.time_axis)
    if data_len < 2:
        raise ValueError("Need at least two time samples to recenter modes")

    intensity = np.sum(np.abs(out.modes_data) ** 2, axis=0)
    i_peak = int(np.argmax(intensity))
    i_target = int(round(peak_target_frac * (data_len - 1)))
    roll_arg = i_target - i_peak

    for ell, emm_list in out.modes_list:
        for emm in emm_list:
            out.set_mode_data(
                ell=ell,
                emm=emm,
                data=np.roll(out.mode(ell, emm), roll_arg),
            )

    if set_peak_time_to_zero:
        dt = float(np.median(np.diff(out.time_axis)))
        out._time_axis = (np.arange(data_len) - i_target) * dt
    else:
        out._time_axis = np.roll(out.time_axis, roll_arg)
    return out


def crop_time_window(wfm_td, t_min=None, t_max=None):
    """Crop a TD ModesArray by its current time axis."""
    out = wfm_td.deepcopy()
    time_axis = out.time_axis
    if t_min is None:
        t_min = time_axis[0]
    if t_max is None:
        t_max = time_axis[-1]

    keep = (time_axis >= t_min) & (time_axis <= t_max)
    idx = np.where(keep)[0]
    if len(idx) == 0:
        raise ValueError(
            f"Requested window [{t_min}, {t_max}] does not overlap "
            f"time range [{time_axis[0]}, {time_axis[-1]}]"
        )
    return out.crop(int(idx[0]), int(idx[-1]) + 1)


def apply_time_taper(wfm_td, taper_width=None, taper_frac=None, sides="both"):
    """Apply a raised-cosine taper to all modes in a TD ModesArray."""
    if sides not in {"both", "left", "right", "none"}:
        message = "sides must be one of 'both', 'left', 'right', 'none'"
        raise ValueError(message)
    if sides == "none":
        return wfm_td.deepcopy()

    out = wfm_td.deepcopy()
    time_axis = out.time_axis
    data_len = len(time_axis)
    if data_len < 3:
        return out

    dt = float(np.median(np.diff(time_axis)))
    duration = float(time_axis[-1] - time_axis[0])
    if taper_width is None:
        if taper_frac is None or taper_frac <= 0:
            return out
        taper_width = taper_frac * duration

    n_taper = int(round(abs(taper_width / dt)))
    n_taper = max(1, min(n_taper, data_len // 2))
    window = np.ones(data_len)
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_taper)))

    if sides in {"both", "left"}:
        window[:n_taper] *= ramp
    if sides in {"both", "right"}:
        window[-n_taper:] *= ramp[::-1]

    for ell, emm_list in out.modes_list:
        for emm in emm_list:
            out.set_mode_data(
                ell=ell,
                emm=emm,
                data=out.mode(ell, emm) * window,
            )
    return out


def prepare_physical_td_window(
    wfm_td,
    t_min=None,
    t_max=None,
    peak_target_frac=0.5,
    taper_width=None,
    taper_frac=None,
    taper_sides="both",
    set_peak_time_to_zero=True,
):
    """Center, crop, and optionally taper a TD circular-FFT buffer."""
    out = recenter_modes_at_peak(
        wfm_td,
        peak_target_frac=peak_target_frac,
        set_peak_time_to_zero=set_peak_time_to_zero,
    )
    if t_min is not None or t_max is not None:
        out = crop_time_window(out, t_min=t_min, t_max=t_max)
    if taper_width is not None or taper_frac is not None:
        out = apply_time_taper(
            out,
            taper_width=taper_width,
            taper_frac=taper_frac,
            sides=taper_sides,
        )
    return out


def fd_modes_to_td_physical_window(
    wfm_fd,
    t_min=None,
    t_max=None,
    peak_target_frac=0.5,
    taper_width=None,
    taper_frac=None,
    taper_sides="both",
    undo_warp=False,
    set_peak_time_to_zero=True,
):
    """Convert FD modes to a centered/cropped/tapered TD window."""
    wfm_td = fd_modes_to_td_modes(wfm_fd, undo_warp=undo_warp)
    return prepare_physical_td_window(
        wfm_td,
        t_min=t_min,
        t_max=t_max,
        peak_target_frac=peak_target_frac,
        taper_width=taper_width,
        taper_frac=taper_frac,
        taper_sides=taper_sides,
        set_peak_time_to_zero=set_peak_time_to_zero,
    )
