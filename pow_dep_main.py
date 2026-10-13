#%%

# ── IMPORTS ────────────────────────────────────────────────────────────────
import glob, os
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, r"") #place the folder where the file pow_dep_functions is located

from pow_dep_functions import (
    load_trpl_data,
    get_spot_size_um,
    power_law,
    plot_lifetimes,
    plot_power_law,
    plot_lifetimes_fluence,
)

#%%
# ═══════════════════════════════════════════════════════════════════════════
# PART B  –  List files
# ═══════════════════════════════════════════════════════════════════════════

folder = r"" #place the measurements are located

for f in sorted(glob.glob(os.path.join(folder, '*.npy'))):
    print(f)
for f in sorted(glob.glob(os.path.join(folder, '*.json'))):
    print(f)

#%%
# ═══════════════════════════════════════════════════════════════════════════
# PART C  –  Load, background-subtract, correct power, plot lifetimes
# ═══════════════════════════════════════════════════════════════════════════

select    = 0
data_dict = load_trpl_data(folder, select)

datArr = data_dict['datArr'].copy()
values = data_dict['power_values'].copy()   # W, raw from instrument

# Background subtraction using lowest-power trace
bg_idx   = np.argmin(values)
bg_trace = datArr[bg_idx]
print(f'BG trace index {bg_idx}  (power = {values[bg_idx]*1e9:.3f} nW)')
datArr = np.clip(datArr - bg_trace, 0, None)
datArr = np.delete(datArr, bg_idx, axis=0)
values = np.delete(values, bg_idx)

# Power correction: beam-splitter (10 % to meter, 90 % to sample) + objective (60 %)
powers_uW = values * (0.90 / 0.10) * 0.60 * 1e6

data_dict['datArr']       = datArr
data_dict['power_values'] = powers_uW    # µW from here on

plot_lifetimes(data_dict, powers_uW, save=False, folder=folder)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(powers_uW, 'o-')
ax.set_xlabel('Measurement index', fontsize=14)
ax.set_ylabel('Power (µW)', fontsize=14)
plt.tight_layout()
plt.show()

#%%
# ═══════════════════════════════════════════════════════════════════════════
# PART D  –  Power-law fit vs fluence (selected subset)
# ═══════════════════════════════════════════════════════════════════════════

start = 15
end   = -2

camera_pixel_size_um = 3.45
save_folder          = r"" #place where the analysis is to be saved

CFG = {
    'figure_size'      : (10, 8),
    'font_axes'        : 46,
    'font_type'        : 'Arial',
    'font_ticks'       : 42,
    'axes_line_width'  : 1.0,
    'tick_width'       : 0.8,
    'tick_length_major': 7,
    'tick_direction'   : 'in',
}

summed_counts = datArr.sum(axis=1)

spot_um        = get_spot_size_um(data_dict['metadata'], camera_pixel_size_um)
spot_area_cm2  = np.pi * ((spot_um / 2) * 1e-4) ** 2
pulse_energy_J = (powers_uW * 1e-6) / (data_dict['effective_rate'] * 1e6)
fluence_uJ_cm2 = pulse_energy_J / spot_area_cm2 * 1e6

print(f'Spot area : {spot_area_cm2:.4e} cm²')
print(f'Fluence range (all): {fluence_uJ_cm2.min():.5f} – {fluence_uJ_cm2.max():.5f} µJ/cm²')

sort_idx     = np.argsort(fluence_uJ_cm2)
fluence_sort = fluence_uJ_cm2[sort_idx]
counts_sort  = summed_counts[sort_idx]

fluence_sel = fluence_sort[start:end]
counts_sel  = counts_sort[start:end]
print(f'Selected {len(fluence_sel)} points  (start={start}, end={end})')
print(f'Fluence range (sel): {fluence_sel.min():.5f} – {fluence_sel.max():.5f} µJ/cm²')

log_x       = np.log(fluence_sel)
log_y       = np.log(counts_sel)
coeffs, cov = np.polyfit(log_x, log_y, 1, cov=True)
beta        = coeffs[0]
a_fit       = np.exp(coeffs[1])
beta_err    = np.sqrt(cov[0, 0])
r2 = 1 - np.sum((log_y - np.polyval(coeffs, log_x))**2) / \
         np.sum((log_y - np.mean(log_y))**2)
print(f'β = {beta:.3f} ± {beta_err:.3f}   R² = {r2:.5f}')

plot_power_law(fluence_sel, counts_sel, a_fit, beta, beta_err,
               label='CTF Quin Py', save=True,
               folder=save_folder, filename='CTF_Quin_Py_powerlaw', CFG=CFG)


#%%
# ═══════════════════════════════════════════════════════════════════════════
# PART E  –  Lifetimes for selected fluence range
# ═══════════════════════════════════════════════════════════════════════════

datArr_sel = datArr[sort_idx][start:end]
print(f'Part E: {datArr_sel.shape[0]} traces  '
      f'(fluence {fluence_sel.min():.5f} – {fluence_sel.max():.5f} µJ/cm²)')

plot_lifetimes_fluence(data_dict, datArr_sel, fluence_sel,
                       label='CTF Quin Py', save=True,
                       folder=save_folder,
                       filename='CTF_Quin_Py_Lifetimes_fluence', CFG=CFG)
# %%
