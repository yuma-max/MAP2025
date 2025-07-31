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
  'Gadget4SubfindHDFCatalogue',
  'ArepoSubfindHDFCatalogue',
  'TNGSubfindHDFCatalogue']


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_radial_distribution(sim_num, snapshots_to_analyze, infopath = "/home/takeichi/MAP/Code_Yuma_2025/Code/Datafiles/Data100_new_edited_Z.csv"):
    """
    Plot the radial distribution of the stars at the designated snapshot for designated simulation.
    The radial distribution plots are following plots
    - Cumulative Stellar Mass
    - r^2 * Density
    - Fraction of Mass inside the sphere
    - Fraction of Mass at each shell。

    Args:
        sim_num (int): number of the simulation we want to take a look at
        snapshots_to_analyze (list): snapshot we want to analyze
        infopath (str): path to the csv file which contains infomation of the simulated halos
    """
    # --- Setting the path ---
    # base_data_folder: where the csv file of the motion of the each star particles are saved
    # info_data_path: 
    base_data_folder = f"/home/takeichi/MAP/Results/{sim_num}"
    info_data_path = f"/home/takeichi/MAP/Code_Yuma_2025/Code/Datafiles/main_halo_com_{sim_num}.csv"
    
    # get halo_ids from the datafolder
    try:
        halo_ids = [d for d in os.listdir(base_data_folder) if os.path.isdir(os.path.join(base_data_folder, d))]
        # Exclude directories which start from "COM_motion" or "r".
        halo_ids = [hid for hid in halo_ids if not hid.startswith("COM") and not hid.startswith("r") and not hid.startswith('.')]
        print(f"Successfully found {len(halo_ids)} halos in {base_data_folder}")
    except FileNotFoundError:
        print(f"Error: Directory not found at {base_data_folder}")
        return
        
    output_dir = os.path.join(base_data_folder, "r_dist_4_panel_plots")
    os.makedirs(output_dir, exist_ok=True)

    try:
        info_df = pd.read_csv(infopath)
    except Exception as e:
        print(f"Fail to read info file. Detail: {e}")
        info_df = None

    print(halo_ids)
    infall_time_map = {}
    if info_df is not None:
        for item in halo_ids:
            try:
                snapshot_at_infall = int(item[:6])
                halo_id_at_infall = int(item[7:])

                full_id = f"{sim_num}{snapshot_at_infall:04d}{halo_id_at_infall}"
                selected = info_df[info_df['ID'] == int(full_id)]
                if not selected.empty:
                    infall_time = selected["time_infall"].iloc[0]
                    infall_time_map[item] = infall_time
            except (ValueError, IndexError):
                print(f"Could not parse halo_id: {item}")

    cmap, norm = None, None
    if infall_time_map:
        min_infall_time = min(infall_time_map.values())
        max_infall_time = max(infall_time_map.values())
        cmap = plt.get_cmap('plasma')
        norm = mcolors.Normalize(vmin=min_infall_time, vmax=max_infall_time)


    for snapshot_str in snapshots_to_analyze:
        SNAPSHOT_TO_ANALYZE = int(snapshot_str)
        print(f"\n--- Starting analysis for Snapshot: {SNAPSHOT_TO_ANALYZE} ---")

        all_r_arrays = []
        all_mass_arrays = []
        labels = []
        colors_for_plot = []
        linestyles = []


        survive_data_folder = "/home/takeichi/MAP/Code_Yuma_2025/Code/Datafiles"
        main_csv_path = os.path.join(survive_data_folder, f"main_halo_com_{sim_num}.csv")
        try:
            main_csv_df = pd.read_csv(main_csv_path)
            main_csv_snap = main_csv_df[main_csv_df["snapshot"] == SNAPSHOT_TO_ANALYZE]
            x_center = main_csv_snap["0"].iloc[0]
            y_center = main_csv_snap["1"].iloc[0]
            z_center = main_csv_snap["2"].iloc[0]
            print(x_center, y_center, z_center)
        except (FileNotFoundError, IndexError, KeyError) as e:
            print(f"Could not get main halo COM for snapshot {SNAPSHOT_TO_ANALYZE}. Skipping. Details: {e}")
            continue


            
        for item in halo_ids:
            snapshot_at_infall = int(item[:6])
            halo_id_at_infall = int(item[7:])
            print(f"Looking at halo{halo_id_at_infall}")
            file_name = f"{snapshot_at_infall:06d}{halo_id_at_infall}"
            data_folder = os.path.join(base_data_folder, item)
            csv_path = os.path.join(data_folder, f"tracked_star_particles{file_name}.csv")
            survive_csv_path = os.path.join(base_data_folder, f"{item}/halo_star_dist{file_name}.csv")

            
            try:
                survive_df = pd.read_csv(survive_csv_path)    
                sub_survive = survive_df[(survive_df["halo_id"] == halo_id_at_infall) & (survive_df["snapshot"] == SNAPSHOT_TO_ANALYZE)]

                if not sub_survive.empty:
                    x = sub_survive["particle_x"].to_numpy()
                    y = sub_survive["particle_y"].to_numpy()
                    z = sub_survive["particle_z"].to_numpy()
                    mass = sub_survive["particle_mass"].to_numpy()
                    r = np.sqrt((x - x_center)**2 + (y - y_center)**2 + (z - z_center)**2)
                    all_r_arrays.append(r)
                    all_mass_arrays.append(mass)
                    labels.append(f"ID {halo_id_at_infall} (survived)")
                    if cmap and norm and item in infall_time_map:
                        colors_for_plot.append(cmap(norm(infall_time_map[item])))
                    else:
                        colors_for_plot.append("gray") 
                    #colors_for_plot.append("purple")
                    linestyles.append("-")


            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"--> File loading error for {csv_path}: {e}")
                continue

            try:
                df = pd.read_csv(csv_path)   
                sub = df[df["snapshot"] == SNAPSHOT_TO_ANALYZE]
                if not sub.empty:
                    r = np.sqrt((sub['x'] - x_center)**2 + (sub['y'] - y_center)**2 + (sub['z'] - z_center)**2)
                    all_r_arrays.append(r.values)
                    all_mass_arrays.append(sub['mass'].values)
                    labels.append(f"ID {halo_id_at_infall} (Infall {snapshot_at_infall})")
                    if cmap and norm and item in infall_time_map:
                        colors_for_plot.append(cmap(norm(infall_time_map[item])))
                    else:
                        colors_for_plot.append("gray") 
                    linestyles.append("--")
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"--> File loading error for {csv_path}: {e}")
                continue

            


        if not all_r_arrays:
            print(f"--> No data found to plot for snapshot {SNAPSHOT_TO_ANALYZE}. Plot was not created.")
            continue

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 32), sharex=True)
        fig.suptitle(f"Radial Distribution of Stars (Snapshot {SNAPSHOT_TO_ANALYZE}, Sim {sim_num})", fontsize=20, y=0.95)


        all_r_flat = np.concatenate(all_r_arrays)
        valid_mask = all_r_flat > 0
        if not np.any(valid_mask):
            print("No valid star distances > 0 found. Cannot create plot.")
            plt.close(fig)
            continue
        
        width = 0.5
        r_max = np.max(all_r_flat[valid_mask])
        bins = np.arange(0, r_max + width, width)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        total_mass_in_bins = np.zeros(len(bin_centers))
        all_cumulative_masses = []
        individual_mass_in_bins = []


        for r_array, mass_array in zip(all_r_arrays, all_mass_arrays):
            mass_in_bins, _ = np.histogram(r_array, bins=bins, weights=mass_array)
            total_mass_in_bins += mass_in_bins
            individual_mass_in_bins.append(mass_in_bins)
            all_cumulative_masses.append(np.cumsum(mass_in_bins))


        shell_volumes = (4/3) * np.pi * (bins[1:]**3 - bins[:-1]**3)
        total_cumulative_mass = np.cumsum(total_mass_in_bins)
        total_density = np.divide(total_mass_in_bins, shell_volumes, out=np.zeros_like(total_mass_in_bins, dtype=float), where=shell_volumes!=0)
        total_r2_density = bin_centers**2 * total_density

        for i in range(len(all_r_arrays)):
            # ax1: Cumulative Mass
            ax1.plot(bin_centers, all_cumulative_masses[i], label=labels[i], color=colors_for_plot[i], linestyle=linestyles[i])
            # ax2: r^2 * Density
            density = np.divide(individual_mass_in_bins[i], shell_volumes, out=np.zeros_like(individual_mass_in_bins[i], dtype=float), where=shell_volumes!=0)
            r2_density = bin_centers**2 * density
            ax2.plot(bin_centers, r2_density, label=labels[i], color=colors_for_plot[i], linestyle=linestyles[i])
            # ax3: Cumulative Mass Fraction
            fraction = np.divide(all_cumulative_masses[i], total_cumulative_mass, out=np.zeros_like(all_cumulative_masses[i], dtype=float), where=total_cumulative_mass!=0)
            ax3.plot(bin_centers, fraction, label=labels[i], color=colors_for_plot[i], linestyle=linestyles[i])
            # ax4: Shell Mass Fraction
            fraction1 = np.divide(individual_mass_in_bins[i], total_mass_in_bins, out=np.zeros_like(individual_mass_in_bins[i], dtype=float), where=total_mass_in_bins!=0)
            ax4.plot(bin_centers, fraction1, label=labels[i], color=colors_for_plot[i], linestyle=linestyles[i])

        ax1.plot(bin_centers, total_cumulative_mass, color='black', linewidth=3, linestyle='--', label='Total Accreted')
        ax2.plot(bin_centers, total_r2_density, color='black', linewidth=3, label='Total Accreted')


        ax1.set_yscale('log')
        ax1.set_ylabel(r"Cumulative Stellar Mass [$M_{\odot}$]", fontsize=14)
        ax1.legend(loc='lower right', title="Progenitor Halo", fontsize='small')
        ax1.grid(True, which="both", ls="--", alpha=0.5)

        ax2.set_yscale('log')
        ax2.set_ylabel(r"$r^2 \rho_{\star}$ [$M_{\odot} / \mathrm{kpc}$]", fontsize=14)
        ax2.legend(loc='upper left', title="Progenitor Halo", fontsize='small')
        ax2.grid(True, which="both", ls="--", alpha=0.5)

        ax3.set_ylabel("Fraction of Mass inside the sphere", fontsize=14)
        ax3.set_ylim(0, 1.05)
        ax3.legend(loc='upper left', title="Progenitor Halo", fontsize='small')
        ax3.grid(True, which="both", ls="--", alpha=0.5)

        ax4.set_ylabel("Fraction of Mass at each shell", fontsize=14)
        ax4.set_ylim(0, 1.05)
        ax4.legend(loc='upper left', title="Progenitor Halo", fontsize='small')
        ax4.grid(True, which="both", ls="--", alpha=0.5)
        
        plt.xlabel("Distance from Main Halo COM (kpc)", fontsize=14)
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xscale('log')
            if np.any(valid_mask):
                ax.set_xlim(left=np.min(bin_centers[bin_centers>0])*0.9, right=bin_centers[-1]*1.1)

        if cmap and norm:
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label('Infall Time [$Gyr$]')

        filename = f"r_dist_4-panel_infalltime_snap{SNAPSHOT_TO_ANALYZE:06d}.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        print(f"--> 4-panel plot saved successfully to: {output_path}")

    print("\n--- All snapshots processed. ---")



if __name__ == '__main__':

    SIM_NUM = 329
    
    SNAPSHOTS = ['004096']
    
    INFO_PATH = "/home/takeichi/MAP/Code_Yuma_2025/Code/Datafiles/Data100_new_edited_Z.csv"

    plot_radial_distribution(
        sim_num=SIM_NUM,
        snapshots_to_analyze=SNAPSHOTS,
        infopath=INFO_PATH
    )






