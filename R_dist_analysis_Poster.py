

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


import pynbody as pb
import numpy as np
import pandas as pd
import glob
import os
import h5py
import time
import tables
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
from natsort import natsorted
import importlib.util
import sys
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import ast
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

os.chdir('/home/takeichi/MAP/halo_tracing_code_Nithun')
import halo_trace as ht
from halo_trace import tracing
file_path = '/home/takeichi/MAP/Code_Yuma_2025/Code/star_trace_Yuma.py'
module_name = 'star_trace'

spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

pb.config['halo-class-priority'] = ['HaloNumberCatalogue', 'AHFCatalogue',
  'AmigaGrpCatalogue',
  'VelociraptorCatalogue',
  'SubFindHDFHaloCatalogue',
  'RockstarCatalogue',
  'SubfindCatalogue',
  'NewAdaptaHOPCatalogue',
  'NewAdaptaHOPCatalogueFullyLongInts',
  'AdaptaHOPCatalogue',
  'HOPCatalogue',
  'Gadget4SubfindHDFHaloCatalogue',
  'ArepoSubfindHDFCatalogue',
  'TNGSubfindHDFCatalogue']



def plot_shell_fraction_quad_cmap(sim_num_list, snapshots_to_analyze, infopath = "/home/takeichi/MAP/Code_Yuma_2025/Code/Datafiles/Data100_new_edited_Z.csv"):
    """
    """
    fig, axes = plt.subplots(1, 2, figsize=(25, 10), sharey=True)
    fig.suptitle("Fraction of Mass at Each Shell", fontsize=20, x=0.5, ha='center')
    for i, sim_num in enumerate(sim_num_list):
        base_data_folder = f"/home/takeichi/MAP/Results/{sim_num}"
        output_dir = os.path.join(base_data_folder, "r_dist_quad_cmap_plots")
        os.makedirs(output_dir, exist_ok=True)

        try:
            halo_ids = [d for d in os.listdir(base_data_folder) if os.path.isdir(os.path.join(base_data_folder, d)) and not d.startswith(("COM", "r", "."))]
            info_df = pd.read_csv(infopath)
            print(f"Successfully found {len(halo_ids)} halos and info file.")
        except (FileNotFoundError, Exception) as e:
            print(f"Error reading initial files: {e}")
            return

        for snapshot_str in snapshots_to_analyze:
            SNAPSHOT_TO_ANALYZE = int(snapshot_str)
            print(f"\n--- Starting analysis for Snapshot: {SNAPSHOT_TO_ANALYZE} ---")

            main_csv_path = f"/home/takeichi/MAP/Code_Yuma_2025/Code/Datafiles/main_halo_com_{sim_num}.csv"
            try:
                main_csv_df = pd.read_csv(main_csv_path)
                main_csv_snap = main_csv_df[main_csv_df["snapshot"] == SNAPSHOT_TO_ANALYZE]
                x_center, y_center, z_center = main_csv_snap[["0", "1", "2"]].iloc[0]
            except (FileNotFoundError, IndexError, KeyError) as e:
                print(f"Could not get main halo COM for snapshot {SNAPSHOT_TO_ANALYZE}. Skipping. Details: {e}")
                continue

            survived_halos = []
            disrupted_halos = []
            all_r_values = []

            for item in halo_ids:
                try:
                    snapshot_at_infall = int(item[:6])
                    halo_id_at_infall = int(item[7:])
                    full_id = f"{sim_num}{snapshot_at_infall:04d}{halo_id_at_infall}"
                    selected = info_df[info_df['ID'] == int(full_id)]
                    if selected.empty: continue
                    infall_time = selected["time_infall"].iloc[0]
                    halo_data = {'infall_time': infall_time}
                    file_name = f"{snapshot_at_infall:06d}{halo_id_at_infall}"
                    survive_csv_path = os.path.join(base_data_folder, item, f"halo_star_dist{file_name}.csv")
                    is_surviving = False
                    try:
                        survive_df = pd.read_csv(survive_csv_path)
                        sub_survive = survive_df[survive_df["snapshot"] == SNAPSHOT_TO_ANALYZE]
                        if not sub_survive.empty:
                            x, y, z = sub_survive[["particle_x", "particle_y", "particle_z"]].values.T
                            mass = sub_survive["particle_mass"].to_numpy()
                            is_surviving = True
                    except FileNotFoundError:
                        pass
                    if not is_surviving:
                        csv_path = os.path.join(base_data_folder, item, f"tracked_star_particles{file_name}.csv")
                        df = pd.read_csv(csv_path)
                        sub = df[df["snapshot"] == SNAPSHOT_TO_ANALYZE]
                        if sub.empty: continue
                        x, y, z = sub[['x', 'y', 'z']].values.T
                        mass = sub['mass'].values
                    r = np.sqrt((x - x_center)**2 + (y - y_center)**2 + (z - z_center)**2)
                    all_r_values.extend(r)
                    halo_data.update({'r': r, 'mass': mass})
                    if is_surviving:
                        survived_halos.append(halo_data)
                    else:
                        disrupted_halos.append(halo_data)
                except Exception as e:
                    print(f"Error processing halo {item}: {e}")

            if not survived_halos and not disrupted_halos:
                print(f"No data to plot for snapshot {SNAPSHOT_TO_ANALYZE}.")
                continue

            def get_norm(data, log=False):
                if not data or (log and min(data) <= 0):
                    return None
                min_val, max_val = min(data), max(data)
                if log:
                    return mcolors.LogNorm(vmin=min_val, vmax=max_val)
                return mcolors.Normalize(vmin=min_val, vmax=max_val)

            norm_stime = get_norm([h['infall_time'] for h in survived_halos])
            norm_dtime = get_norm([h['infall_time'] for h in disrupted_halos])
        
            cmap_surv, cmap_disr = plt.get_cmap('winter'), plt.get_cmap('spring_r')

            width = 1
            r_max_val = np.max(all_r_values) if all_r_values else 100
            bins = np.arange(0, r_max_val + width, width)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            total_mass_in_bins = np.zeros(len(bin_centers))
            for halo in survived_halos + disrupted_halos:
                mass_in_bins, _ = np.histogram(halo['r'], bins=bins, weights=halo['mass'])
                total_mass_in_bins += mass_in_bins
                halo['mass_in_bins'] = mass_in_bins

        
        
            # Surviving
            for halo in survived_halos:
                fraction = np.divide(halo['mass_in_bins'], total_mass_in_bins, out=np.zeros_like(total_mass_in_bins, dtype=float), where=total_mass_in_bins!=0)
                if norm_stime:
                    axes[i].plot(bin_centers, fraction, color=cmap_surv(norm_stime(halo['infall_time'])))
            # Disrupted 
            for halo in disrupted_halos:
                fraction = np.divide(halo['mass_in_bins'], total_mass_in_bins, out=np.zeros_like(total_mass_in_bins, dtype=float), where=total_mass_in_bins!=0)
                if norm_dtime:
                    axes[i].plot(bin_centers, fraction, color=cmap_disr(norm_dtime(halo['infall_time'])))

            # --- Setting Plots ---
            axes[i].set_xscale('log') 
            axes[i].set_ylim(0, 1.05)
            axes[i].set_xlabel("Radial Distance (kpc)", fontsize=14)
            axes[i].grid(True, which="both", ls="--", alpha=0.5)
            axes[0].set_ylabel("Fraction of Mass at each shell", fontsize=14)
            if sim_num == 329:
                name="Elena"
            if sim_num == 229:
                name="Sonia"
            axes[i].set_title(f"{name}, z=0", fontsize=16)

       
            def add_colorbars(ax, norm_s, norm_d, cmap_s, cmap_d,label_s, label_d):
                if not (norm_s and norm_d): return
                divider = make_axes_locatable(ax)
                cax_s = divider.append_axes("right", size="5%", pad=0.1)
                cax_d = divider.append_axes("right", size="5%", pad=0.7)
                fig.colorbar(cm.ScalarMappable(norm=norm_s, cmap=cmap_s), cax=cax_s, label=label_s)
                fig.colorbar(cm.ScalarMappable(norm=norm_d, cmap=cmap_d), cax=cax_d, label=label_d)

            add_colorbars(axes[i], norm_stime, norm_dtime, cmap_surv, cmap_disr, r"Surviving, Infall Time (Gyr)", r"Disrupted, Infall Mass($M_\odot$)")

    
    filename = f"r_dist_shell_fraction_quad_cmap_snap{SNAPSHOT_TO_ANALYZE:06d}.svg"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path,
        format='svg',
        dpi=300,
        bbox_inches='tight',
        transparent=False)
    plt.show()
    print(f"--> Quad-colormap plot saved successfully to: {output_path}")

    print("\n--- All snapshots processed. ---")


if __name__ == '__main__':
    SIM_NUM = 329
    SNAPSHOTS = ['004096']
    INFO_PATH = "/home/takeichi/MAP/Code_Yuma_2025/Code/Datafiles/Data100_new_edited_Z.csv"

    plot_shell_fraction_quad_cmap(
        sim_num=SIM_NUM,
        snapshots_to_analyze=SNAPSHOTS,
        infopath=INFO_PATH
    )





