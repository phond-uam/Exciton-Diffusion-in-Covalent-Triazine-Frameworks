import numpy as np
import glob
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import wofz
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ── Binning functions ─────────────────────────────────────────────────────────

def ConstantBins(x, c):
    return c

def PowerLawForBins(x, a, b, c):
    return a * x**b + c


# ── Data loading & parameter extraction ──────────────────────────────────────

def LoadData(paths):
    datas = []
    for name in paths:
        datas.append(np.loadtxt(name, delimiter=','))
    return datas


def param_from_metadata(metadata_json):
    TimeResolution = metadata_json['TimeResolution']
    AcquisitionTime = metadata_json['AcquisitionTime']
    Binning = metadata_json['Binning']
    laser_pw = metadata_json['laserPower']
    laser_div = metadata_json['laserDivider']
    objective = metadata_json['objective']
    steps = metadata_json['steps']
    MaxRepRate = metadata_json['MaxRepRate'] * 1e6

    try:
        picoharp_300 = metadata_json['picoharp_300']
    except KeyError:
        picoharp_300 = True

    if picoharp_300:
        if MaxRepRate == 80e6:
            NPulsesSeen = int(8 / laser_div) if laser_div < 8 else 1
        else:
            NPulsesSeen = 1
        lens_magnification = 3.636363633333333
    else:
        NPulsesSeen = 1
        lens_magnification = 3.68686869

    xmin = metadata_json['x_range'][0]
    xmax = metadata_json['x_range'][1]
    ymin = metadata_json['y_range'][0]
    ymax = metadata_json['y_range'][1]

    APD_range = np.linalg.norm(np.array([xmin, ymin]) - np.array([xmax, ymax]))
    total_magnification = lens_magnification * objective
    sample_range = APD_range / total_magnification
    SpatialResolution = sample_range / steps

    return TimeResolution, AcquisitionTime, Binning, laser_pw, laser_div, NPulsesSeen, xmin, xmax, ymin, ymax, SpatialResolution, MaxRepRate


def extract_parameters(path_param):
    with open(path_param[0], "r") as f:
        params = f.read()

    TimeResolution = float(params[params.find('_ns_') - 5:params.find('_ns_')])
    AcquisitionTime = float(params[params.find('_ns_') + 20:params.find('_ms_')])
    Binning = int(params[params.find('_laserPower_') - 1:params.find('_laserPower_')])
    laser_pw = float(params[params.find('laserPower_') + 12:params.find('_uW_')])
    laser_div = int(params[params.find('_steps_') - 1:params.find('_steps_')])
    range_x = params[params.find('_x_range_') + 9:params.find('_y_range_') - 3]
    range_y = params[params.find('_y_range_') + 9:params.find('_mm_objective_')]
    steps = int(params[params.find('_steps_') + 7:params.find('_x_range_')])
    objective = params[params.find('_objective_') + 11:params.find('_objective_') + 14]
    objective = int(objective.replace('x', ''))

    MaxRepRate = 1e6
    NPulsesSeen = int(8 / laser_div) if laser_div != 1 else 7

    xmin = float(range_x[1:range_x.find(',')])
    xmax = float(range_x[range_x.find(',') + 1:-1])
    ymin = float(range_y[1:range_y.find(',')])
    ymax = float(range_y[range_y.find(',') + 1:-1])

    APD_range = np.linalg.norm(np.array([xmin, ymin]) - np.array([xmax, ymax]))
    lens_magnification = 3.636363633333333
    total_magnification = lens_magnification * objective
    sample_range = APD_range / total_magnification
    SpatialResolution = sample_range / steps

    return TimeResolution, AcquisitionTime, Binning, laser_pw, laser_div, NPulsesSeen, xmin, xmax, ymin, ymax, SpatialResolution


def extract_parameters_from_metadata(metadata_path, folder):
    if len(metadata_path) == 0:
        metadata_path = glob.glob(folder + '2*metadata.json')
        with open(metadata_path[0], 'r') as f:
            metadata = json.load(f)
        TimeResolution, AcquisitionTime, Binning, laser_pw, laser_div, NPulsesSeen, xmin, xmax, ymin, ymax, SpatialResolution, MaxRepRate = param_from_metadata(metadata)
    else:
        TimeResolution, AcquisitionTime, Binning, laser_pw, laser_div, NPulsesSeen, xmin, xmax, ymin, ymax, SpatialResolution = extract_parameters(metadata_path)
        MaxRepRate = 80e6
        print('Assuming MaxRepRate = 80 MHz')

    return TimeResolution, AcquisitionTime, Binning, laser_pw, laser_div, NPulsesSeen, xmin, xmax, ymin, ymax, SpatialResolution, MaxRepRate


# ── Signal processing ─────────────────────────────────────────────────────────

def merge_Pulses(data, TimeResolution, Divider, NumberOfPulsesSeen, MaxRepetitionRate=80e6):
    TimeResolution = TimeResolution * 1e-3
    RepRate = MaxRepetitionRate / Divider
    TimeBetweenPulse = (1 / RepRate) * 1e9
    BinsBetweenPulse = int(TimeBetweenPulse / TimeResolution)
    time = np.arange(BinsBetweenPulse) * TimeResolution

    data_merged = []
    for row in data:
        for pulse_nr in range(NumberOfPulsesSeen):
            if pulse_nr == 0:
                counts = row[:BinsBetweenPulse]
            else:
                counts += row[BinsBetweenPulse * pulse_nr:BinsBetweenPulse * (pulse_nr + 1)]
        data_merged.append(counts)

    return data_merged, time


def binningNonlinear(x, function, functionparams, keep_last_pixels=True, returnBinningArray=False):
    binning_array = []
    i = 0
    while np.sum(binning_array) < len(x):
        binning_array.append(int(function(i, *functionparams)))
        i += 1
    binning_array = np.array(binning_array)

    binned_x = []
    if (binning_array < 1).any():
        print('Number of bins need to be a positive integer.')
        return None
    elif (binning_array == 1).all():
        return (x, returnBinningArray) if returnBinningArray else x
    elif binning_array[0] > len(x):
        print('The first binning window is larger than the array.')
        return x
    else:
        for i, element in enumerate(x):
            if sum(binning_array[:i + 1]) >= len(x):
                if keep_last_pixels:
                    binned_x.append(sum(x[sum(binning_array[:i]):]) / len(x[sum(binning_array[:i]):]))
                return (binned_x, binning_array) if returnBinningArray else binned_x
            binned_x.append(sum(x[sum(binning_array[:i]):sum(binning_array[:i + 1])]) / binning_array[i])


def binningNonlinearErrors(x, function, functionparams, keep_last_pixels=True, returnBinningArray=False):
    binning_array = []
    i = 0
    while np.sum(binning_array) < len(x):
        binning_array.append(int(function(i, *functionparams)))
        i += 1
    binning_array = np.array(binning_array)

    binned_x = []
    if (binning_array < 1).any():
        print('Number of bins need to be a positive integer.')
        return None
    elif (binning_array == 1).all():
        return (x, returnBinningArray) if returnBinningArray else x
    elif binning_array[0] > len(x):
        print('The first binning window is larger than the array.')
        return (x, returnBinningArray) if returnBinningArray else x
    else:
        for i, element in enumerate(x):
            if sum(binning_array[:i + 1]) >= len(x):
                if keep_last_pixels:
                    binned_x.append(np.sqrt(sum([v**2 for v in x[sum(binning_array[:i]):]]))
                                    / len(x[sum(binning_array[:i]):]))
                return (binned_x, binning_array) if returnBinningArray else binned_x
            binned_x.append(np.sqrt(sum([v**2 for v in x[sum(binning_array[:i]):sum(binning_array[:i + 1])]]))
                            / binning_array[i])


def Normalize(MergedImage):
    normalizeddata = []
    for dat in MergedImage:
        normalized = np.array(dat) / np.max(dat)
        normalizeddata.append(normalized)
    return normalizeddata


def rawdata_to_readydata(rawData, TimeResolution, laser_div, NPulsesSeen, MaxRepRate, binning_time_function, binning_time_params):
    rawData_2 = [matrix[1:] for matrix in rawData]
    DataArr = np.array(rawData_2)

    dat_OnePeak_All = []
    for data in DataArr:
        dat_OnePeak_unordered, _ = merge_Pulses(data, TimeResolution, laser_div, NPulsesSeen, MaxRepRate)
        max_pos = np.argmax(sum(dat_OnePeak_unordered))
        dat_OnePeak_ordered = np.roll(dat_OnePeak_unordered, -max_pos, axis=1)
        dat_OnePeak_All.append(dat_OnePeak_ordered)

    dataBinned = []
    for data in dat_OnePeak_All:
        dataBinned.append(binningNonlinear(data.transpose(), binning_time_function, binning_time_params, keep_last_pixels=True))

    dataBinnedNorm = []
    for data in dataBinned:
        dataBinnedNorm.append(np.array(Normalize(data)))

    return dat_OnePeak_All, dataBinned, dataBinnedNorm


def get_spatial_time_values(dat_OnePeak_All, TimeResolution, spatialResolution, binning_time_function, binning_time_params):
    TimeResolution = TimeResolution * 1e-3
    spatialValues = np.arange(dat_OnePeak_All[0].shape[0]) * spatialResolution
    spatialValues = spatialValues - (spatialValues[-1] - spatialValues[0]) / 2
    timeValuesUncut, BinningArrayTimeUncut = binningNonlinear(
        np.arange(dat_OnePeak_All[0].shape[1]) * TimeResolution,
        binning_time_function, binning_time_params,
        keep_last_pixels=True, returnBinningArray=True
    )
    timeValuesUncut = timeValuesUncut - min(timeValuesUncut)
    return spatialValues, timeValuesUncut, BinningArrayTimeUncut


# ── FWHM ──────────────────────────────────────────────────────────────────────

def fwhm(p1, xp1=[-1], averaging='average', factor=0.5):
    def interpolate(x, xp, fp):
        return fp[0] + (x - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])

    if len(xp1) == 1:
        xp1 = [i for i in range(len(p1))]

    maximum, max_index = max([p1[i], i] for i in range(len(p1)))
    goal = maximum * factor
    lefts, rights = [], []

    for i, value in enumerate(p1[:-1]):
        if p1[i] <= goal and p1[i + 1] > goal:
            (lefts if i < max_index else rights).append(
                interpolate(goal, [p1[i], p1[i + 1]], [xp1[i], xp1[i + 1]]))
        elif p1[i] >= goal and p1[i + 1] < goal:
            (lefts if i < max_index else rights).append(
                interpolate(goal, [p1[i], p1[i + 1]], [xp1[i], xp1[i + 1]]))

    left = np.median(lefts) if averaging == 'median' else np.mean(lefts)
    right = np.median(rights) if averaging == 'median' else np.mean(rights)
    return abs(right - left)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_all_measurements(dataBinnedNorm, spatialValues, spatialResolution, timeValuesUncut, saveFig=False, folder=None):
    for i, array in enumerate(dataBinnedNorm):
        plt.figure()
        plt.title(str(i))
        plt.pcolormesh(spatialValues, timeValuesUncut, array, cmap="plasma")
        plt.gca().invert_yaxis()
        plt.xlabel('distance (um)')
        plt.ylabel('time (ns)')
        if saveFig:
            plt.savefig(folder + str(i) + '_spectrum.png')

    plt.figure()
    for i, array in enumerate(dataBinnedNorm):
        plt.title('Profiles at t = 0')
        plt.plot((np.arange(len(array[0])) - (len(array[0]) - 1) / 2) * spatialResolution,
                 array[0], label=i)
        plt.xlabel('distance (um)')
        plt.ylabel('normalized counts (a.u.)')
        plt.legend(loc='best')
        if saveFig:
            plt.savefig(folder + str(i) + 'profile_t0.png')
        print(fwhm(array[0]) * spatialResolution * 1e3)

def get_rising_edge(dat_OnePeak_All, arraySelection, TimeResolution, window_ns=1.0, rebin_factor=5):
    '''
    Recovers the true pre-peak rise of the pulse from the wrapped tail of the
    peak-aligned raw traces in dat_OnePeak_All. For DISPLAY only -- not fit.

    IN:
        dat_OnePeak_All : list of arrays (n_spatial, n_raw_time_bins), from
                          rawdata_to_readydata (peak already rolled to index 0)
        arraySelection  : same indices used in merge_selected_arrays
        TimeResolution  : same metadata value (ps) used elsewhere in your code
        window_ns       : how much pre-peak time to recover (ns)
        rebin_factor    : raw bins averaged per displayed point (raw ~TimeResolution
                          ps each; rebin_factor=5 at 4 ps raw -> 20 ps display bins)
    OUT:
        time_rise      : 1D array, negative time values (ns), ascending to 0
        intensity_rise : 1D array, spatially-summed raw counts (unnormalized)
    '''
    TimeResolution_ns = TimeResolution * 1e-3
    n_raw = int(window_ns / TimeResolution_ns)
    traces = []
    for i in arraySelection:
        raw = dat_OnePeak_All[i]            # (n_spatial, n_time), peak at index 0
        tail = raw[:, -n_raw:]               # wrapped-around pre-peak segment
        traces.append(tail.sum(axis=0))      # spatial sum
    avg_trace = np.mean(traces, axis=0)      # average across selected scans

    n_out = len(avg_trace) // rebin_factor
    rebinned = avg_trace[:n_out*rebin_factor].reshape(n_out, rebin_factor).mean(axis=1)

    dt_display = TimeResolution_ns * rebin_factor
    time_rise = -dt_display * np.arange(len(rebinned), 0, -1)
    return time_rise, rebinned


# ── Merging ───────────────────────────────────────────────────────────────────

def merge_selected_arrays(dataBinned, arraySelection, cutData_time, timeValuesUncut, BinningArrayTimeUncut, spatialResolution):
    cutData = [(np.abs(timeValuesUncut - t)).argmin() for t in cutData_time]

    min_shape = 200
    dat_to_merge = []
    for i in arraySelection:
        uncentered_dat = np.array(dataBinned[i][cutData[0]:cutData[1]])
        averaged_first = np.mean(uncentered_dat[0:5, :], axis=0)
        max_index = np.argmax(averaged_first)
        remove_value = max_index * 2 - uncentered_dat.shape[1]

        if remove_value > 0:
            centered_dat = np.delete(uncentered_dat, range(np.abs(remove_value)), axis=1)
        elif remove_value < 0:
            centered_dat = np.delete(uncentered_dat, range(np.abs(remove_value)) + max_index * 2, axis=1)
        else:
            centered_dat = uncentered_dat

        min_shape = np.min([centered_dat.shape[1], min_shape])
        dat_to_merge.append(centered_dat)

    dat_ready_merge = []
    for i in range(len(dat_to_merge)):
        current_shape = dat_to_merge[i].shape[1]
        if current_shape > min_shape:
            cut_col = (current_shape - min_shape) // 2
            dat_ready_merge.append(dat_to_merge[i][:, cut_col:-cut_col])
        dat_ready_merge.append(dat_to_merge[i][:, :min_shape])

    Merged = np.array(dat_ready_merge).mean(axis=0)

    MergedNormalized = np.array([row / np.max(row) for row in Merged])

    spatialValues = (np.arange(Merged.shape[1]) - np.argmax(sum(Merged))) * spatialResolution
    timeValuesMerged = timeValuesUncut[cutData[0]:cutData[1]]
    timeValuesMerged = timeValuesMerged - min(timeValuesMerged)
    BinningArrayTime = BinningArrayTimeUncut[cutData[0]:cutData[1]]

    return Merged, MergedNormalized, spatialValues, timeValuesMerged, BinningArrayTime


# ── Lifetime fitting ──────────────────────────────────────────────────────────

def plot_merge_lifetimes_fit(Merged, timeValuesMerged, timeSlice,
                             dat_OnePeak_All, arraySelection, TimeResolution,  
                             saveFig=False, folder=None, img=True, side=None,
                             poisson_weighted=True):
    
    """
    Bi-exponential lifetime fit on the spatially summed decay.
 
    Fit model (normalised):
        I(t) = A1*exp(-t/tau1) + A2*exp(-t/tau2) + offset
        tau2  = tau1 + delta_tau,  delta_tau >= 0
 
    Works correctly on any dataset — initial guess is derived from the data,
    bounds scale with Tspan so they never silently clamp the fit.
    """
 
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
 
    eps = 1e-6
 
    # ── 1. Build 1D decay: spatial sum over the merged map ───────────────────
    DataSummed = np.sum(Merged, axis=1).astype(float)
    t          = np.asarray(timeValuesMerged, float)
 
    m        = np.isfinite(DataSummed) & np.isfinite(t) & (DataSummed > 0)
    t_fit    = t[m]
    y_counts = DataSummed[m]
    y_max    = max(float(np.nanmax(y_counts)), 1e-12)
    y_norm   = y_counts / y_max
 
    Tspan = max(float(t_fit[-1] - t_fit[0]), 1e-12)
 
    # ── 2. Model ──────────────────────────────────────────────────────────────
    def double_exp_constrained(t, A1, A2, tau1, delta_tau, offset):
        tau2 = tau1 + np.maximum(delta_tau, 1e-12)
        return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + offset
 
    # ── 3. Initial guess ────────────────────
    inv_e_idx      = np.argmin(np.abs(y_norm - 1.0 / np.e))
    tau1_guess     = max(float(t_fit[inv_e_idx]), 1e-3)
    delta_tau_guess = max(tau1_guess * 4.0, Tspan / 10.0)
    offset_guess   = max(float(np.percentile(y_norm, 2)) * 0.5, 1e-5)
 
    p0 = [
        0.75,             # A1  — dominant fast component
        0.20,             # A2  — slower component
        tau1_guess,       # tau1 estimated from 1/e point of data
        delta_tau_guess,  # delta_tau = tau2 - tau1
        offset_guess,     # offset
    ]
 
    # ── 4. Bounds — scaled to Tspan so they work on any dataset ──────────────
    bounds = (
        [0.0,  0.0,  1e-3,         1e-3,         0.0   ],
        [2.0,  2.0,  Tspan * 0.5,  Tspan * 2.0,  0.05  ]
    )
 
    print(f"  tau1 guess : {tau1_guess:.3f} ns")
    print(f"  tau2 guess : {tau1_guess + delta_tau_guess:.3f} ns")
    print(f"  bounds tau1: [1e-3, {Tspan*0.5:.2f}] ns")
    print(f"  bounds tau2: [1e-3, {Tspan*2.0:.2f}] ns")
 
    # ── 5. Log-weighted sigma — tail fitted as well as peak ──────────────────
    # sigma = y_norm minimises (y-fit)²/y² = relative residuals
    # equivalent to fitting in log-scale with equal weight at every point
    sigma_log = np.clip(y_norm, eps, None)
 
    popt_d, pcov_d = curve_fit(
        double_exp_constrained,
        t_fit, y_norm,
        p0     = p0,
        bounds = bounds,
        sigma  = sigma_log,
        absolute_sigma = False,
        maxfev = 80000
    )
 
    # ── 6. Extract lifetimes ──────────────────────────────────────────────────
    A1, A2, tau1, dtau, offset = popt_d
    tau2    = tau1 + dtau
    tau_eff = (A1 * tau1 + A2 * tau2) / max(A1 + A2, 1e-12)
 
    perr     = np.sqrt(np.diag(pcov_d))
    tau1_err = perr[2]
    tau2_err = np.sqrt(perr[2]**2 + perr[3]**2)
    S = A1 + A2
    J = np.array([
        -A2*dtau/S**2,   # d(tau_eff)/dA1
        A1*dtau/S**2,   # d(tau_eff)/dA2
        1.0,            # d(tau_eff)/dtau1
        A2/S,           # d(tau_eff)/d(delta_tau)
        0.0             # d(tau_eff)/d(offset)
    ])
    tau_eff_err = np.sqrt(J @ pcov_d @ J)
 
    y_fit     = double_exp_constrained(t_fit, *popt_d)
    residuals = (y_norm - y_fit) / np.clip(y_norm, eps, None)
 
    print(f"\n  τ1     = {tau1:.3f} ± {tau1_err:.3f} ns   (A1 = {A1:.3f})")
    print(f"  τ2     = {tau2:.3f} ± {tau2_err:.3f} ns   (A2 = {A2:.3f})")
    print(f"  offset = {offset:.5f}")
    print(f"  τ_eff  = {tau_eff:.3f} ± {tau_eff_err:.3f} ns   (amplitude-weighted)")

    time_rise, intensity_rise = get_rising_edge(dat_OnePeak_All, arraySelection, TimeResolution)
    y_rise_norm = intensity_rise / y_max    
 
    # ── 7. Plot ───────────────────────────────────────────────────────────────
    if img:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
 
        data_style = dict(
            marker='o', linestyle='None', ms=5,
            mfc='none', mec='royalblue', mew=1.3,
            label='Data'
        )
 
        annotation = (
            f"$\\tau_1$ = {tau1:.2f} ± {tau1_err:.2f} ns\n"
            f"$\\tau_2$ = {tau2:.2f} ± {tau2_err:.2f} ns\n"
            f"$\\tau_{{\\rm eff}}$ = {tau_eff:.2f} ns"
        )
 
        # linear
        ax[0].plot(time_rise, y_rise_norm, marker='o', linestyle='None', ms=5,
               mfc='none', mec='royalblue', mew=1.0, label='Rise (not fit)')
        ax[0].plot(t_fit, y_norm, **data_style)
        ax[0].plot(t_fit, y_fit, '--', lw=2, color='orangered', label='Bi-exp fit')
        ax[0].set_xlabel('Time (ns)')
        ax[0].set_ylabel('Intensity (normalised)')
        ax[0].set_title('Lifetime trace' + (f' — {side}' if side else ''))
        ax[0].text(0.38, 0.95, annotation,
                   transform=ax[0].transAxes, fontsize=11,
                   va='top', linespacing=1.7,
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='grey', alpha=0.7))
        ax[0].legend()
 
        # log-linear — primary check for tail quality
        ax[1].semilogy(time_rise, np.clip(y_rise_norm, eps, None), marker='o', linestyle='None', ms=5,
               mfc='none', mec='royalblue', mew=1.0, label='Rise (not fit)')
        ax[1].semilogy(t_fit, np.clip(y_norm, eps, None), **data_style)
        ax[1].semilogy(t_fit, np.clip(y_fit,  eps, None),
                       '--', lw=2, color='orangered', label='Bi-exp fit')
        ax[1].set_xlabel('Time (ns)')
        ax[1].set_title('Log-linear')
        ax[1].legend()

        # log-linear — primary check for tail quality
        ax[2].loglog(t_fit, np.clip(y_norm, eps, None), **data_style)
        ax[2].loglog(t_fit, np.clip(y_fit,  eps, None),
                       '--', lw=2, color='orangered', label='Bi-exp fit')
        ax[2].set_xlabel('Time (ns)')
        ax[2].set_title('Log-linear')
        ax[2].legend()
 
        # # relative residuals — flat noise = good fit
        # ax[2].axhline(0,     color='grey', lw=1.0, linestyle='--')
        # ax[2].axhline( 0.2,  color='grey', lw=0.7, linestyle=':')
        # ax[2].axhline(-0.2,  color='grey', lw=0.7, linestyle=':')
        # ax[2].plot(t_fit, residuals, marker='o', ms=4,
        #            mfc='none', mec='royalblue', mew=1.1, linestyle='None')
        # ax[2].set_xlabel('Time (ns)')
        # ax[2].set_ylabel('(data − fit) / data')
        # ax[2].set_title('Relative residuals')
        # ax[2].set_ylim(-1, 1)
 
        for a in ax.flatten():
            for spine in a.spines.values():
                spine.set_linewidth(1.3)
            a.tick_params(which='both', direction='in', top=True, right=True)
 
        plt.tight_layout()
        plt.show()
 
    # ── 8. Return ─────────────────────────────────────────────────────────────
    fit_data = {
    "t_fit"       : t_fit,
    "y_norm"      : y_norm,
    "sigma_norm"  : sigma_log,
    "time_rise"   : time_rise,      # add this
    "y_rise_norm" : y_rise_norm,    # add this
    }
 
    return DataSummed, [popt_d], [pcov_d], fit_data



# ── Voigt / diffusion map ─────────────────────────────────────────────────────

def Voigt(x, height, center, sigma, gamma):
    return height * np.real(wofz(((x - center) + 1j * gamma) / sigma / np.sqrt(2))) / sigma / np.sqrt(2 * np.pi)


def plot_merge_diffusion_map(Merged, spatialValues, timeValuesMerged, BinningArrayTime, arraySelection,
                              SpatialResolution, cut_time, saveFig=False, folder=None, img=True, side=None,
                              save_path=None, save_file_prefix=None):

    def VoigtReduced2dNonlinearBinning(x, center, gamma, offset, *args):
        heights = args[:int(len(args) / 2)]
        sigmas = args[int(len(args) / 2):]
        _, heights = np.meshgrid([0] * x.shape[1], heights)
        _, sigmas = np.meshgrid([0] * x.shape[1], sigmas)
        _, offsets = np.meshgrid([0] * x.shape[1], offset * BinningArrayTime)
        return (Voigt(x, heights, center, sigmas, gamma) + offsets).ravel()

    xOffset = SpatialResolution / 2

    CFG = {
        "font_axes": 28, "font_ticks": 26, "font_legend": 18,
        "font_style": "Arial", "axes_line_width": 1.2,
        "tick_width": 1.5, "tick_length_major": 7,
        "tick_direction": "in", "tick_pad": 10,
    }

    def apply_style(ax, font_axes=None, font_ticks=None):
        fa = font_axes or CFG["font_axes"]
        ft = font_ticks or CFG["font_ticks"]
        for spine in ax.spines.values():
            spine.set_linewidth(CFG["axes_line_width"])
        ax.tick_params(axis='both', which='major', direction=CFG["tick_direction"],
                       width=CFG["tick_width"], length=CFG["tick_length_major"],
                       labelsize=ft, pad=CFG["tick_pad"], top=False, right=False)
        ax.minorticks_off()

    if img:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        mesh = ax.imshow(
            Normalize(Merged - Merged.min() * 0), aspect='auto', origin='lower',
            extent=[spatialValues[0] - xOffset, spatialValues[-1] - xOffset,
                    timeValuesMerged[0], timeValuesMerged[-1]],
            cmap="plasma"
        )
        ax.set_ylim(0, cut_time)
        ax.set_xlim(-1.5, 1.5)
        ax.set_xlabel(r'Position ($\mu$m)', fontsize=CFG["font_axes"], labelpad=10)
        ax.set_ylabel('Time (ns)', fontsize=CFG["font_axes"], labelpad=10)
        ax.set_xticks([-1.5, 0, 1.5])
        ax.set_yticks([0, 10, 20])
        apply_style(ax)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_ticks([0, 1])
        cbar.set_label('Intensity (a.u.)', fontsize=CFG["font_axes"], labelpad=2, rotation=270, va='bottom')
        cbar.ax.tick_params(labelsize=CFG["font_ticks"], width=CFG["tick_width"],
                            length=CFG["tick_length_major"], direction=CFG["tick_direction"])
        for spine in cbar.ax.spines.values():
            spine.set_linewidth(CFG["axes_line_width"])

        if save_path and save_file_prefix:
            for fmt in ("svg", "pdf"):
                fig.savefig(os.path.join(save_path, f"{save_file_prefix}_Diffusion_Map.{fmt}"),
                            format=fmt, bbox_inches="tight")
        plt.tight_layout()
        plt.show()

    _, BinningArrayTime2D = np.meshgrid([0] * Merged.shape[1], BinningArrayTime)
    DataToFit = Merged * BinningArrayTime2D
    X, _ = np.meshgrid(spatialValues, [0] * Merged.shape[0])

    lowBounds = [-100000] + [0] + [0] + [0] * Merged.shape[0] * 2
    highBounds = [10000000] + [1000000000] + [1000000000] + [10000000] * Merged.shape[0] * 2
    guess = [0, 0.1, 0, *np.amax(DataToFit, 1), *([0.3] * DataToFit.shape[0])]
    popt, success = curve_fit(VoigtReduced2dNonlinearBinning, X, DataToFit.ravel(), guess,
                               bounds=(lowBounds, highBounds))

    if img:
        CFG_large = {**CFG, "font_axes": 52, "font_ticks": 50, "font_legend": 40}
        custom_colors = ['royalblue', 'orange', 'limegreen', 'red', 'purple']
        fineSpatialValues = np.linspace(spatialValues[0], spatialValues[-1], 100)
        fineX, _ = np.meshgrid(fineSpatialValues, [0] * DataToFit.shape[0])

        fig, ax = plt.subplots(figsize=(12.5, 11.15), dpi=300)

        for i, time_temp in enumerate([0, 0.2, 1]):
            row = (np.abs(timeValuesMerged - time_temp)).argmin()
            norm = VoigtReduced2dNonlinearBinning(fineX, *popt).reshape(
                (DataToFit.shape[0], len(fineSpatialValues)))[row].max()
            ax.scatter(spatialValues - popt[0], DataToFit[row] / norm,
                       facecolors='none', edgecolors=custom_colors[i],
                       label=str(np.round(timeValuesMerged[row], 1)) + ' ns',
                       s=450, linewidths=2, zorder=5)
            ax.plot(fineSpatialValues - popt[0],
                    VoigtReduced2dNonlinearBinning(fineX, *popt).reshape(
                        (DataToFit.shape[0], len(fineSpatialValues)))[row] / norm,
                    color=custom_colors[i], lw=3, zorder=4)

        row0 = (np.abs(timeValuesMerged - 0.0)).argmin()
        n_rows = DataToFit.shape[0]
        sigma0 = popt[3 + n_rows:3 + 2 * n_rows][row0]
        y_sigma = 0.28
        ax.annotate("", xy=(-sigma0 / 2, y_sigma), xytext=(-sigma0 * 2.0, y_sigma),
                    arrowprops=dict(arrowstyle="<-", lw=6.5, color="black"))
        ax.annotate("", xy=(+sigma0 * 2.0, y_sigma), xytext=(+sigma0 / 2, y_sigma),
                    arrowprops=dict(arrowstyle="->", lw=6.5, color="black"))
        ax.text(0, y_sigma, r"$\sigma$", ha="center", va="center", fontsize=50, color="black")

        ax.set_xlabel(r'Position ($\mu$m)', fontsize=CFG_large["font_axes"], labelpad=10)
        ax.set_ylabel('Intensity (a.u.)', fontsize=CFG_large["font_axes"], labelpad=10)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.05, 1.15)
        ax.set_xticks([-1.5, 0, 1.5])
        ax.set_yticks([0, 0.5, 1])
        apply_style(ax, font_axes=CFG_large["font_axes"], font_ticks=CFG_large["font_ticks"])
        ax.legend(frameon=False, loc="upper right",
                  bbox_to_anchor=(1, 0.99),
                  prop={'size': CFG_large["font_legend"], 'family': CFG_large["font_style"]})

        ax_inset = inset_axes(ax, width="30%", height="35%", loc="upper left",
                              bbox_to_anchor=(0.054, -0.02, 1, 1), bbox_transform=ax.transAxes)
        for i, time_temp in enumerate([0, 0.2, 1]):
            row = (np.abs(timeValuesMerged - time_temp)).argmin()
            norm = VoigtReduced2dNonlinearBinning(fineX, *popt).reshape(
                (DataToFit.shape[0], len(fineSpatialValues)))[row].max()
            ax_inset.scatter(spatialValues - popt[0], DataToFit[row] / norm,
                             facecolors='none', edgecolors=custom_colors[i], s=220, linewidths=2, zorder=5)
            ax_inset.plot(fineSpatialValues - popt[0],
                          VoigtReduced2dNonlinearBinning(fineX, *popt).reshape(
                              (DataToFit.shape[0], len(fineSpatialValues)))[row] / norm,
                          color=custom_colors[i], lw=1.9, zorder=5)
        ax_inset.set_xlim(0.5, 1)
        ax_inset.set_ylim(-0.03, 0.5)
        ax_inset.set_xticks([0.5, 1])
        ax_inset.set_yticks([0, 0.5])
        ax_inset.tick_params(labelsize=24, width=1.5, top=False, right=False)
        for spine in ax_inset.spines.values():
            spine.set_linewidth(1.2)

                # ── Standalone inset plot (zoomed spatial profiles, full-figure style) ──
        fig_inset, ax_inset2 = plt.subplots(figsize=(12.5, 11.15), dpi=300)

        for i, time_temp in enumerate([0, 0.2, 1]):
            row = (np.abs(timeValuesMerged - time_temp)).argmin()
            norm = VoigtReduced2dNonlinearBinning(fineX, *popt).reshape(
                (DataToFit.shape[0], len(fineSpatialValues)))[row].max()
            ax_inset2.scatter(spatialValues - popt[0], DataToFit[row] / norm,
                              facecolors='none', edgecolors=custom_colors[i],
                              label=str(np.round(timeValuesMerged[row], 1)) + ' ns',
                              s=450, linewidths=2, zorder=5)
            ax_inset2.plot(fineSpatialValues - popt[0],
                          VoigtReduced2dNonlinearBinning(fineX, *popt).reshape(
                              (DataToFit.shape[0], len(fineSpatialValues)))[row] / norm,
                          color=custom_colors[i], lw=3, zorder=4)

        ax_inset2.set_xlim(0.5, 1)
        ax_inset2.set_ylim(-0.03, 0.5)
        ax_inset2.set_xticks([0.5, 1])
        ax_inset2.set_yticks([0, 0.5])
        ax_inset2.set_xlabel(r'Position ($\mu$m)', fontsize=CFG_large["font_axes"], labelpad=10)
        ax_inset2.set_ylabel('Intensity (a.u.)', fontsize=CFG_large["font_axes"], labelpad=10)
        apply_style(ax_inset2, font_axes=CFG_large["font_axes"], font_ticks=CFG_large["font_ticks"])
        ax_inset2.legend(frameon=False, loc="upper right",
                         prop={'size': CFG_large["font_legend"], 'family': CFG_large["font_style"]})

        ax_inset2.text(0.03, 0.03, save_file_prefix,
                       transform=ax_inset2.transAxes,
                       fontsize=CFG_large["font_axes"],
                       fontfamily=CFG_large["font_style"],
                       ha="left", va="bottom")
        plt.tight_layout()

        if save_path and save_file_prefix:
            for fmt in ("svg", "pdf"):
                fig_inset.savefig(os.path.join(save_path, f"{save_file_prefix}_Linecut_inset_standalone.{fmt}"),
                                  format=fmt, bbox_inches="tight")
        plt.show()

        
        
        plt.tight_layout()

        if save_path and save_file_prefix:
            for fmt in ("svg", "pdf"):
                fig_inset.savefig(os.path.join(save_path, f"{save_file_prefix}_Linecut_inset_standalone.{fmt}"),
                                  format=fmt, bbox_inches="tight")
        plt.show()


    return popt.copy(), success.copy()


# ── MSD & diffusivity ─────────────────────────────────────────────────────────

def extract_MSD(MergedNormalized, timeValuesMerged, popt, success, cutOff, time_fit,
                binningSigmasFunction, binningSigmas, t1_index=0):
    sigmas, sigmasBinningArray = binningNonlinear(popt[-MergedNormalized.shape[0] + cutOff:],
                                                   binningSigmasFunction, binningSigmas,
                                                   returnBinningArray=True)
    sigmas = np.array(sigmas)
    timeValues = np.array(binningNonlinear(timeValuesMerged[cutOff:], binningSigmasFunction, binningSigmas))
    timeValues = timeValues - min(timeValues)
    tend_index = (np.abs(timeValues - time_fit)).argmin()

    deltaSigmaSquare = sigmas**2 - sigmas[0]**2

    error = np.sqrt(np.diag(success))[-MergedNormalized.shape[0] + cutOff:]
    error = np.array(binningNonlinearErrors(error, binningSigmasFunction, binningSigmas))
    errorsquare = abs(2 * sigmas * error)
    errorsquare0 = errorsquare[t1_index]
    errorsquaresum = np.array(np.sqrt(errorsquare**2 + errorsquare0**2))
    errorsquaresum[t1_index] = errorsquare0

    return timeValues, tend_index, deltaSigmaSquare, errorsquare, errorsquaresum, sigmas


def extract_diffusivity(timeValues, deltaSigmaSquare, errorsquare, t1_index, tend_index):
    def diffusion(t, D, alpha):
        t = np.array(t)
        return 2 * D * (t - timeValues[t1_index])**1 + deltaSigmaSquare[t1_index]

    guessDiffusion = [0.001, 1]
    lowBounds = [-1, 0]
    highBounds = [10000000, 2 + 1e-10]
    pDiffusion, successDiffusion = curve_fit(
        diffusion, timeValues[t1_index:tend_index], deltaSigmaSquare[t1_index:tend_index],
        guessDiffusion, errorsquare[t1_index:tend_index], bounds=(lowBounds, highBounds)
    )
    errorDiffusion = np.sqrt(np.diag(successDiffusion))
    return pDiffusion, successDiffusion, errorDiffusion


def save_msd_plot_data(timeValues, deltaSigmaSquare, errorsquaresum, pDiffusion, errorDiffusion,
                        time_arr_fit, fit_values, arraySelection, folder=None, sample_name=None):
    from datetime import datetime
    if folder is None:
        folder = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    max_length = max(len(timeValues), len(time_arr_fit))
    time_exp = np.full(max_length, np.nan)
    msd_exp = np.full(max_length, np.nan)
    msd_error = np.full(max_length, np.nan)
    time_fit = np.full(max_length, np.nan)
    msd_fit = np.full(max_length, np.nan)

    time_exp[:len(timeValues)] = timeValues
    msd_exp[:len(deltaSigmaSquare)] = deltaSigmaSquare
    msd_error[:len(errorsquaresum)] = errorsquaresum
    time_fit[:len(time_arr_fit)] = time_arr_fit
    msd_fit[:len(fit_values)] = fit_values

    df = pd.DataFrame({'time_ns': time_exp, 'msd_um2': msd_exp,
                       'msd_error_um2': msd_error, 'time_fit_ns': time_fit, 'msd_fit_um2': msd_fit})

    base = sample_name or f'msd_plot_data_{arraySelection}_{timestamp}'
    params_base = (f'{sample_name}_msd_fit_parameters'
                   if sample_name else f'msd_fit_parameters_{arraySelection}_{timestamp}')

    msd_plot_file = os.path.join(folder, f'{base}_msd_plot_data.csv')
    df.to_csv(msd_plot_file, index=False)

    params_df = pd.DataFrame({
        'Parameter': ['D (cm²/s)', 'D_error (cm²/s)', 'alpha', 'alpha_error'],
        'Value': [pDiffusion[0] * 1e1, errorDiffusion[0] * 1e1, pDiffusion[1], errorDiffusion[1]]
    })
    params_file = os.path.join(folder, f'{params_base}.csv')
    params_df.to_csv(params_file, index=False)

    print(f"✓ Saved MSD plot data to: {msd_plot_file}")
    print(f"✓ Saved fit parameters to: {params_file}")
    return msd_plot_file, params_file


def plot_MSD_and_Diff(timeValues, deltaSigmaSquare, errorsquaresum, pDiffusion, errorDiffusion,
                       arraySelection, cutOff, t1_index, tend_index, time_fit,
                       saveFig=False, folder=None, figsize=(8, 6), xlim=None, ylim=None,
                       sample_name=None, save_path=None, save_file_prefix=None):

    CFG = {
        "font_axes": 24, "font_ticks": 22, "font_legend": 18,
        "axes_line_width": 1.2, "tick_width": 1.2,
        "tick_length_major": 7, "tick_direction": "in", "tick_pad": 5,
    }

    def apply_style(ax):
        for spine in ax.spines.values():
            spine.set_linewidth(CFG["axes_line_width"])
        ax.tick_params(axis='both', which='major', direction=CFG["tick_direction"],
                       width=CFG["tick_width"], length=CFG["tick_length_major"],
                       labelsize=CFG["font_ticks"], pad=CFG["tick_pad"], top=False, right=False)
        ax.minorticks_off()

    def diffusion(t, D, alpha):
        return 2 * D * (np.array(t) - timeValues[t1_index])**1 + deltaSigmaSquare[t1_index]

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6), dpi=300)

    ax.errorbar(timeValues, deltaSigmaSquare, yerr=errorsquaresum,
                fmt='o', markerfacecolor='None', linestyle='None',
                markeredgecolor='royalblue', markeredgewidth=1.7, alpha=0.7, zorder=1, markersize=9)
    ax.errorbar(timeValues[t1_index:tend_index], deltaSigmaSquare[t1_index:tend_index],
                yerr=errorsquaresum[t1_index:tend_index],
                fmt='o', markerfacecolor='None', linestyle='None',
                markeredgecolor='orangered', markeredgewidth=1.7, alpha=0.7, markersize=9)

    time_arr_fit = timeValues[t1_index:tend_index + tend_index // 5]
    fit_values = diffusion(time_arr_fit, *pDiffusion)
    ax.plot(time_arr_fit, fit_values, zorder=10, linestyle='dashed', color='red')

    ax.set_ylabel(r'$\sigma(t)^2 - \sigma(0)^2$ ($\mu$m²)', fontsize=CFG["font_axes"], labelpad=10)
    ax.set_xlabel('Time (ns)', fontsize=CFG["font_axes"], labelpad=10)
    apply_style(ax)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # ax.set_yticks([0, 0.5, 1.0, 1.5])

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    ax.text(0.03, 0.80,
            r"D (cm$^2$/s) : %.3f $\pm$ %.3f" "\n" r"$\alpha$ : %.2f $\pm$ %.2f" % (
                pDiffusion[0] * 1e1, errorDiffusion[0] * 1e1, pDiffusion[1], errorDiffusion[1]),
            fontsize=CFG["font_legend"], color='black',
            horizontalalignment='left', verticalalignment='baseline',
            transform=ax.transAxes, linespacing=1.8)

    plt.tight_layout()

    if save_path and save_file_prefix:
        for fmt in ("png", "svg", "pdf"):
            fig.savefig(os.path.join(save_path, f"{save_file_prefix}_diffusivity.{fmt}"),
                        format=fmt, dpi=300, bbox_inches='tight')

    save_msd_plot_data(timeValues, deltaSigmaSquare, errorsquaresum, pDiffusion, errorDiffusion,
                       time_arr_fit, fit_values, arraySelection,
                       folder=save_path, sample_name=sample_name)

    plt.show()
    return time_arr_fit, fit_values
