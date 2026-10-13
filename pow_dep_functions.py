import json, os, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter


def _compute_time_axis(RepRate, TimeResolution, laser_divider, histogram_size):
    """Returns (effective_rate MHz, period_ns, time_windw bins, Time_arr ns)."""
    effective_rate = RepRate / laser_divider
    period_ns      = 1000.0 / effective_rate
    time_windw     = int(1e6 / (effective_rate * TimeResolution))
    Time_arr       = np.arange(histogram_size) * TimeResolution * 1e-3
    return effective_rate, period_ns, time_windw, Time_arr


def load_trpl_data(folder, select=0):
    npy_files  = sorted(f for f in os.listdir(folder) if f.endswith('.npy'))
    json_files = sorted(f for f in os.listdir(folder) if f.endswith('.json'))

    if not npy_files:
        raise FileNotFoundError('No .npy files found.')
    if select >= len(npy_files):
        raise IndexError(f'select={select} out of range ({len(npy_files)} files).')

    base      = os.path.splitext(npy_files[select])[0]
    ts_match  = re.search(r'(\d+)', base)
    if not ts_match:
        raise ValueError(f'No timestamp in filename: {base}')
    timestamp = ts_match.group(1)

    matched_json = next((j for j in json_files if timestamp in j), None)
    if matched_json is None:
        raise FileNotFoundError(f'No .json with timestamp {timestamp}.')

    print(f'  Data : {npy_files[select]}')
    print(f'  Meta : {matched_json}')

    data_raw = np.load(os.path.join(folder, npy_files[select]),
                       allow_pickle=True).item()
    with open(os.path.join(folder, matched_json)) as f:
        metadata = json.load(f)

    RepRate        = metadata.get('Rep_rate (MHz)', metadata.get('RepRate (MHz)', 80))
    TimeResolution = metadata.get('Time resolution (ps)', 8)
    laser_divider  = metadata.get('laser_divider', 1)
    histogram_size = data_raw['datapoints'].shape[1]

    effective_rate, period_ns, time_windw, Time_arr = _compute_time_axis(
        RepRate, TimeResolution, laser_divider, histogram_size)

    print(f'  RepRate={RepRate} MHz  divider={laser_divider}  '
          f'eff={effective_rate} MHz  res={TimeResolution} ps  '
          f'bins/period={time_windw}  traces={data_raw["datapoints"].shape[0]}')

    return {
        'positions'     : data_raw['positions'],
        'power_values'  : data_raw['power_values'],
        'datArr'        : data_raw['datapoints'],
        'Time_arr'      : Time_arr,
        'histogram_size': histogram_size,
        'time_windw'    : time_windw,
        'period_ns'     : period_ns,
        'effective_rate': effective_rate,
        'RepRate'       : RepRate,
        'TimeResolution': TimeResolution,
        'laser_divider' : laser_divider,
        'AcqTime'       : metadata.get('Acquisition time (ms)'),
        'spot_size_um'  : metadata.get('Spot_size (diam, um)'),
        'spot_size_px'  : metadata.get('Spot_size (diam, px)'),
        'objective'     : metadata.get('objective (X)'),
        'picoharp'      : metadata.get('Picoharp used'),
        'metadata'      : metadata,
    }


def get_spot_size_um(metadata, camera_pixel_size_um=3.45):
    magnification = metadata['objective (X)']
    spot_px       = metadata['Spot_size (diam, px)']
    spot_um       = spot_px * camera_pixel_size_um / magnification
    print(f'  Spot: {spot_px} px  obj={magnification}X  → {spot_um:.4f} µm')
    return spot_um


def power_law(x, a, b):
    return a * x**b


def _apply_style(ax, font_type='Arial', font_axes=18, font_ticks=14,
                 lw=1.5, tick_width=1.2, tick_length=5, tick_direction='in'):
    for spine in ax.spines.values():
        spine.set_linewidth(lw)
    ax.tick_params(axis='both', which='major',
                   labelsize=font_ticks, width=tick_width,
                   length=tick_length, direction=tick_direction,
                   top=False, right=False)
    ax.tick_params(axis='both', which='minor', top=False, right=False,
                   left=False, bottom=False)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
        lbl.set_fontfamily(font_type)


def _shift_and_normalize(curve, Time_arr, time_windw):
    """Peak-shift, normalize, return (t_shift, h_norm)."""
    h        = curve[:time_windw].astype(float)
    peak_idx = np.argmax(h)
    t_shift  = Time_arr[:len(h)] - Time_arr[peak_idx]
    h_norm   = h / np.max(h)
    return t_shift, h_norm


def plot_lifetimes(data_dict, powers_uW, size=(7, 5),
                   save=False, folder='', filename='lifetimes'):
    datArr     = data_dict['datArr']
    Time_arr   = data_dict['Time_arr']
    time_windw = data_dict['time_windw']
    period_ns  = data_dict['period_ns']

    fig, ax = plt.subplots(figsize=size)
    cmap    = plt.cm.plasma
    colors  = cmap(np.linspace(0, 1, len(powers_uW)))

    plotted = []
    for i, (curve, pwr) in enumerate(zip(datArr, powers_uW)):
        if np.max(curve[:time_windw]) <= 0:
            continue
        t_shift, h_norm = _shift_and_normalize(curve, Time_arr, time_windw)
        mask = (t_shift > 0) & (t_shift <= period_ns) & (h_norm > 0)
        if mask.any():
            ax.loglog(t_shift[mask], h_norm[mask], color=colors[i])
            plotted.append(pwr)

    ax.set_xlabel('Time (ns)', fontsize=18, fontweight='bold',
                  fontfamily='Arial', labelpad=15)
    ax.set_ylabel('Intensity (a.u.)', fontsize=18, fontweight='bold',
                  fontfamily='Arial', labelpad=15)
    _apply_style(ax)

    plotted = np.array(plotted if plotted else powers_uW)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(plotted.min(), plotted.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0, location='top')
    cbar.set_ticks([plotted.min(), plotted.max()])
    cbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    cbar.set_label('Power (μW)', fontsize=14, fontweight='bold', labelpad=1)
    cbar.ax.tick_params(labelsize=12, width=1.2)
    for lbl in cbar.ax.get_xticklabels():
        lbl.set_fontweight('bold')

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(folder, filename + '.svg'),
                    dpi=300, bbox_inches='tight')
    plt.show()


def plot_power_law(fluence, counts, a_fit, beta_fit, beta_err,
                   label='Sample', color='tab:purple',
                   save=False, folder='', filename='powerlaw', CFG=None):
    if CFG is None:
        CFG = {'figure_size': (10, 8), 'font_axes': 46, 'font_type': 'Arial',
               'font_ticks': 42, 'axes_line_width': 1.0, 'tick_width': 0.8,
               'tick_length_major': 7, 'tick_direction': 'in'}

    x_fit = np.linspace(fluence.min(), fluence.max(), 350)
    y_fit = power_law(x_fit, a_fit, beta_fit)

    fig, ax = plt.subplots(figsize=CFG['figure_size'], dpi=350)
    ax.scatter(fluence, counts, marker='o', s=1050,
               facecolors='none', edgecolors=color, linewidths=3,
               zorder=6, label=label)
    ax.plot(x_fit, y_fit, '--', color='red', lw=1.8, zorder=4)
    ax.set_xscale('log'); ax.set_yscale('log')

    ax.set_xlabel('Fluence (µJ/cm²)', fontsize=CFG['font_axes'],
                  fontfamily=CFG['font_type'], labelpad=10)
    ax.set_ylabel('Counts (a.u.)', fontsize=CFG['font_axes'],
                  fontfamily=CFG['font_type'], labelpad=10)
    _apply_style(ax, font_type=CFG['font_type'], font_ticks=CFG['font_ticks'],
                 lw=CFG['axes_line_width'], tick_width=CFG['tick_width'],
                 tick_length=CFG['tick_length_major'],
                 tick_direction=CFG['tick_direction'])
    ax.tick_params(axis='both', pad=10)

    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(-0.06, 0.99),
              prop={'size': 36, 'family': CFG['font_type']},
              markerscale=0.8, handletextpad=0.5)
    ax.text(0.16, 0.83, fr'$\beta$ = {beta_fit:.3f} $\pm$ {beta_err:.3f}',
            transform=ax.transAxes, fontsize=34,
            fontfamily=CFG['font_type'], ha='left', va='top')

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(folder, filename + '.svg'),
                    format='svg', bbox_inches='tight')
        fig.savefig(os.path.join(folder, filename + '.pdf'),
                    format='pdf', bbox_inches='tight')
    plt.show()


def plot_lifetimes_fluence(data_dict, datArr_sel, fluence_sel,
                            label='Sample', save=False,
                            folder='', filename='lifetimes_fluence', CFG=None):
    if CFG is None:
        CFG = {'figure_size': (11, 10), 'font_axes': 46, 'font_type': 'Arial',
               'font_ticks': 42, 'axes_line_width': 1.0, 'tick_width': 0.8,
               'tick_length_major': 7, 'tick_direction': 'in'}

    Time_arr   = data_dict['Time_arr']
    time_windw = data_dict['time_windw']
    period_ns  = data_dict['period_ns']

    fig, ax = plt.subplots(figsize=CFG['figure_size'], dpi=300)
    cmap   = plt.cm.plasma
    colors = cmap(np.linspace(0, 1, len(fluence_sel)))

    plotted = []
    for i, (curve, fl) in enumerate(zip(datArr_sel, fluence_sel)):
        if np.max(curve[:time_windw]) <= 0:
            continue
        t_shift, h_norm = _shift_and_normalize(curve, Time_arr, time_windw)
        mask = (t_shift > 0) & (t_shift <= period_ns) & (h_norm > 0)
        if mask.any():
            ax.plot(t_shift[mask], h_norm[mask], color=colors[i], lw=1.0)
            plotted.append(fl)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Time (ns)', fontsize=CFG['font_axes'],
                  fontfamily=CFG['font_type'], labelpad=10)
    ax.set_ylabel('Intensity (a.u.)', fontsize=CFG['font_axes'],
                  fontfamily=CFG['font_type'], labelpad=10)
    _apply_style(ax, font_type=CFG['font_type'], font_ticks=CFG['font_ticks'],
                 lw=CFG['axes_line_width'], tick_width=CFG['tick_width'],
                 tick_length=CFG['tick_length_major'],
                 tick_direction=CFG['tick_direction'])
    ax.tick_params(axis='both', pad=10)

    plotted = np.array(plotted if plotted else fluence_sel)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(plotted.min(), plotted.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.0)
    cbar.set_label('Fluence (µJ/cm²)', fontsize=CFG['font_axes'],
                   fontfamily=CFG['font_type'], labelpad=20)
    cbar.ax.tick_params(labelsize=CFG['font_ticks'], width=CFG['tick_width'])
    fmt = ScalarFormatter(useMathText=False)
    fmt.set_scientific(True); fmt.set_powerlimits((0, 0))
    cbar.ax.yaxis.set_major_formatter(fmt)
    cbar.ax.yaxis.get_offset_text().set_fontfamily(CFG['font_type'])
    cbar.ax.yaxis.get_offset_text().set_fontsize(CFG['font_ticks'] - 4)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontfamily(CFG['font_type'])

    ax.legend([label], frameon=False, loc='lower left',
              bbox_to_anchor=(0.0, 0.04),
              prop={'size': 39, 'family': CFG['font_type']})

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(folder, filename + '.svg'),
                    format='svg', bbox_inches='tight')
        fig.savefig(os.path.join(folder, filename + '.pdf'),
                    format='pdf', bbox_inches='tight')
    plt.show()
