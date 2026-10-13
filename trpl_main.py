#%%

# ── PART A: Calling the imports

import numpy as np
import glob
import os
import pandas as pd

import sys
sys.path.insert(0, r"/") #place the location of the folder containing trpl_functions file

from trpl_functions import (
    ConstantBins, PowerLawForBins,
    LoadData, extract_parameters_from_metadata,
    rawdata_to_readydata, get_spatial_time_values,
    plot_all_measurements,
    merge_selected_arrays,
    plot_merge_lifetimes_fit,
    plot_merge_diffusion_map,
    extract_MSD, extract_diffusivity, plot_MSD_and_Diff,
)

# ── Configuration ─────────────────────────────────────────────────────────────

folder     = r"/" #place the location of the measurement file to be analysed
save_path  = r"" #place the location of the folder where the analysis to be saved
sample_name = "" #name of the analysed file

binning_time_function = ConstantBins
binning_time_params   = [50]




#%%

# ── PART B: Load & analyse all measurements ───────────────────────────────────

path          = sorted(glob.glob(folder + '2*.csv'))
metadata_path = glob.glob(folder + '2*meas_params.txt')

for i, fname in enumerate(path):
    print(f"{i}: {os.path.basename(fname)}")

(TimeResolution, AcquisitionTime, Binning,
 laser_pw, laser_div, NPulsesSeen,
 xmin, xmax, ymin, ymax,
 SpatialResolution, MaxRepRate) = extract_parameters_from_metadata(metadata_path, folder)

print('Parameters extracted: analysing measurements...')

rawData = LoadData(path)
dat_OnePeak_All, dataBinned, dataBinnedNorm = rawdata_to_readydata(
    rawData, TimeResolution, laser_div, NPulsesSeen, MaxRepRate,
    binning_time_function, binning_time_params
)

spatialValues, timeValuesUncut, BinningArrayTimeUncut = get_spatial_time_values(
    dat_OnePeak_All, TimeResolution, SpatialResolution,
    binning_time_function, binning_time_params
)

plot_all_measurements(dataBinnedNorm, spatialValues * 1e3, SpatialResolution,
                      timeValuesUncut, saveFig=False, folder=folder)

#%%

# ── PART C: Merge selected data & fit ─────────────────────────────────────────

arraySelection = []
cutData_time   = [0, 20]  # ns

Merged, MergedNormalized, spatialValues, timeValuesMerged, BinningArrayTime = merge_selected_arrays(
    dataBinned, arraySelection, cutData_time,
    timeValuesUncut, BinningArrayTimeUncut, SpatialResolution
)

DataSummed, lifetime_fit, lifetime_pcov, fit_data = plot_merge_lifetimes_fit(
    Merged, timeValuesMerged, timeSlice=[4, 0],
    dat_OnePeak_All=dat_OnePeak_All,    
    arraySelection=arraySelection,
    TimeResolution=TimeResolution,
    saveFig=False, folder=folder
)

# --- Recompute fit parameters and uncertainties directly from what plot_merge_lifetimes_fit returned ---
A1, A2, tau1, dtau, offset = lifetime_fit[0]
pcov_d = lifetime_pcov[0]
tau2 = tau1 + dtau

perr = np.sqrt(np.diag(pcov_d))
tau1_err = perr[2]
tau2_err = np.sqrt(perr[2]**2 + perr[3]**2)

tau_eff = (A1 * tau1 + A2 * tau2) / max(A1 + A2, 1e-12)

S = A1 + A2
J = np.array([
    -A2*dtau/S**2,
     A1*dtau/S**2,
     1.0,
     A2/S,
     0.0
])

tau_eff_err = np.sqrt(max(J @ pcov_d @ J, 0.0))

# --- Save fit input data ---

os.makedirs(save_path, exist_ok=True)

max_len = max(len(fit_data["t_fit"]), len(fit_data["time_rise"]))

def pad(arr, n):
    out = np.full(n, np.nan)
    out[:len(arr)] = arr
    return out

df_fit = pd.DataFrame({
    "t_ns_decay":       pad(fit_data["t_fit"], max_len),
    "y_norm_decay":     pad(fit_data["y_norm"], max_len),
    "sigma_norm_decay": pad(fit_data["sigma_norm"], max_len),
    "t_ns_rise":        pad(fit_data["time_rise"], max_len),
    "y_norm_rise":      pad(fit_data["y_rise_norm"], max_len),
})

df_fit.to_csv(os.path.join(save_path, f"{sample_name}_Lifetimes_fit_input_data.csv"), index=False)

# --- Save fit parameters (with uncertainties) ---
with open(os.path.join(save_path, f"{sample_name}_lifetime_fit_parameters.txt"), "w", encoding="utf-8") as f:
    f.write("Lifetime fit parameters\n-----------------------\n")
    f.write(f"tau1:   {tau1:.6f} +- {tau1_err:.6f} ns\n")
    f.write(f"tau2:   {tau2:.6f} +- {tau2_err:.6f} ns\n")
    f.write(f"A1:     {A1:.6f}\n")
    f.write(f"A2:     {A2:.6f}\n")
    f.write(f"offset: {offset:.6f}\n\n")
    f.write("Amplitude-weighted effective lifetime\n--------------------------------------\n")
    f.write(f"tau_eff: {tau_eff:.6f} +- {tau_eff_err:.6f} ns\n")


popt, success = plot_merge_diffusion_map(
    Merged, spatialValues * 1e3, timeValuesMerged, BinningArrayTime,
    arraySelection, SpatialResolution,
    cut_time=cutData_time[1] - 1,
    saveFig=False, folder=folder,
    save_path=save_path, save_file_prefix=sample_name
)

#%%

# ── PART D: MSD & diffusivity ─────────────────────────────────────────────────

binningSigmasFunction = PowerLawForBins
binningSigmas         = [0, 1, 1]
cutOff                = 0
t1_index              = 0
time_fit              = 0.8  # ns

xlim = [-0.1, 2]
ylim = [-0.002, 0.05]

timeValues, tend_index, deltaSigmaSquare, errorsquare, errorsquaresum, sigmas = extract_MSD(
    MergedNormalized, timeValuesMerged, popt, success,
    cutOff, time_fit, binningSigmasFunction, binningSigmas, t1_index=t1_index
)

pDiffusion, successDiffusion, errorDiffusion = extract_diffusivity(
    timeValues, deltaSigmaSquare, errorsquare, t1_index, tend_index
)

timeValues_fit, fit_values = plot_MSD_and_Diff(
    timeValues, deltaSigmaSquare, errorsquaresum,
    pDiffusion, errorDiffusion, arraySelection,
    cutOff, t1_index, tend_index, time_fit,
    saveFig=False, folder=folder,
    xlim=xlim, ylim=ylim,
    sample_name=sample_name,
    save_path=save_path, save_file_prefix=sample_name
)

print(f"\nDiffusion coefficient:")
print(f"  D = {pDiffusion[0] * 1e1:.4f} ± {errorDiffusion[0] * 1e1:.4f} cm²/s")

#%%

# ── PART E: Diffusion length ──────────────────────────────────────────────────────────

A1_, A2_, tau1_, dtau_, offset_ = lifetime_fit[0]
pcov_d = lifetime_pcov[0]          # full covariance matrix, already in your namespace
tau2_ = tau1_ + dtau_
tau_eff = (A1_ * tau1_ + A2_ * tau2_) / max(A1_ + A2_, 1e-12)

# --- tau_eff error via delta method (same Jacobian as discussed earlier) ---
S = A1_ + A2_
J = np.array([
    -A2_*dtau_/S**2,   # d(tau_eff)/dA1
     A1_*dtau_/S**2,   # d(tau_eff)/dA2
     1.0,               # d(tau_eff)/dtau1
     A2_/S,             # d(tau_eff)/d(delta_tau)
     0.0                # d(tau_eff)/d(offset)
])
tau_eff_err = np.sqrt(max(J @ pcov_d @ J, 0.0))

# --- diffusion values ---
D = pDiffusion[0] * 1e1              # cm^2/s
D_err = errorDiffusion[0] * 1e1      # cm^2/s
tau_eff_s = tau_eff * 1e-9           # s
tau_eff_err_s = tau_eff_err * 1e-9   # s

L_cm = np.sqrt(2 * D * tau_eff_s)
L_nm = L_cm * 1e7

# --- error propagation ---
rel_err_D   = D_err / D
rel_err_tau = tau_eff_err_s / tau_eff_s
rel_err_L   = 0.5 * np.sqrt(rel_err_D**2 + rel_err_tau**2)
L_err_nm = L_nm * rel_err_L

print("\nDiffusion length:")
print(f"L = {L_nm:.2f} ± {L_err_nm:.2f} nm")

os.makedirs(save_path, exist_ok=True)

with open(os.path.join(save_path, f"{sample_name}_diffusion_length.txt"), "w") as f:
    f.write(f"L = {L_nm:.2f} nm\n")

# %%
