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

def plot_r_vs_infallproperty(sim_num_list, snapshots_to_analyze, infopath = "/home/takeichi/MAP/Code_Yuma_2025/Code/Datafiles/Data100_new_edited_Z.csv"):
    """
    Plot the scatter plot of every halos in the simulation I mentioned.
    Plot 1
    x-axis: infall time
    y-axis: distance between the COM of the main halo and the COM of the stars in sattelite galaxy at snapshot 4096
    change the shape based on the simulation num
    change the line style based on the halo is disruoted or not.
    
    Plot 2
    x-axis: infall mass
    y-axis: distance between the COM of the main halo and the COM of the stars in sattelite galaxy at snapshot 4096
    change the shape based on the simulation num
    change the line style based on the halo is disruoted or not.
    
    Args:
        sim_num (int): number of the simulation we want to take a look at
        snapshots_to_analyze (list): snapshot we want to analyze
        infopath (str): path to the csv file which contains infomation of the simulated halos
    """
    Result_dir = "/home/takeichi/MAP/Results"
    output_dir = os.path.join(Result_dir, "infall_property_vs_r")
    os.makedirs(output_dir, exist_ok = True)
   
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(23, 8))
    fig.suptitle("Properties of progenitor vs. distance from center of mass of main halo", fontsize=20, y=0.95)
    for sim_num in sim_num_list:
        print(f"Looking for simulation h{sim_num}")
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


        try:
            info_df = pd.read_csv(infopath)
        except Exception as e:
            print(f"Fail to read info file. Detail: {e}")
            info_df = None
        # Get the information of the progenitor properties
        infall_mass_list = []
        infall_time_list = []
        simulation_number_list = []
        if info_df is not None:
            for item in halo_ids:
                try:
                    snapshot_at_infall = int(item[:6])
                    halo_id_at_infall = int(item[7:])
                    simulation_number_list.append(sim_num)
                    full_id = f"{sim_num}{snapshot_at_infall:04d}{halo_id_at_infall}"
                    selected = info_df[info_df['ID'] == int(full_id)]
                    if not selected.empty:
                        infall_time = selected["time_infall"].iloc[0]
                        infall_mass = selected["infall_mass"].iloc[0]
                        infall_time_list.append(infall_time)
                        infall_mass_list.append(infall_mass)
                except (ValueError, IndexError):
                    print(f"Could not parse halo_id: {item}")

                    


        snapshots_to_analyze
        print(f"\n--- Starting analysis for Snapshot: {snapshots_to_analyze} ---")

        r_lists = []
        color_list = []
        survive_data_folder = "/home/takeichi/MAP/Code_Yuma_2025/Code/Datafiles"
        main_csv_path = os.path.join(survive_data_folder, f"main_halo_com_{sim_num}.csv")
        try:
            main_csv_df = pd.read_csv(main_csv_path)
            main_csv_snap = main_csv_df[main_csv_df["snapshot"] == snapshots_to_analyze]
            x_center = main_csv_snap["0"].iloc[0]
            y_center = main_csv_snap["1"].iloc[0]
            z_center = main_csv_snap["2"].iloc[0]
            print(x_center, y_center, z_center)
        except (FileNotFoundError, IndexError, KeyError) as e:
            print(f"Could not get main halo COM for snapshot {snapshots_to_analyze}. Skipping. Details: {e}")
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
                sub_survive = survive_df[(survive_df["halo_id"] == halo_id_at_infall) & (survive_df["snapshot"] == snapshots_to_analyze)]

                if not sub_survive.empty:
                    x_com = sub_survive["halo_x_com"].iloc[0]
                    y_com = sub_survive["halo_y_com"].iloc[0]
                    z_com = sub_survive["halo_z_com"].iloc[0]

                    r = np.sqrt(x_com**2+y_com**2+z_com**2)
                    r_lists.append(r)
                    color_list.append("blue")


            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"--> File loading error for {csv_path}: {e}")
                continue

            try:
                df = pd.read_csv(csv_path)   
                sub_disrupt = df[df["snapshot"] == snapshots_to_analyze]
                if not sub_disrupt.empty:
                    mass = sub_disrupt["mass"].to_numpy()
                    M = np.sum(mass)
                    if M > 0:
                        xcom = np.sum(sub_disrupt["x"].to_numpy() * mass) / M
                        ycom = np.sum(sub_disrupt["y"].to_numpy() * mass) / M
                        zcom = np.sum(sub_disrupt["z"].to_numpy() * mass) / M
                        r = np.sqrt((xcom - x_center)**2 + (ycom - y_center)**2 + (zcom - z_center)**2)
                        r_lists.append(r)
                    color_list.append("red")
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"--> File loading error for {csv_path}: {e}")
                continue
        
        infall_time_array = np.array(infall_time_list)
        infall_mass_array = np.array(infall_mass_list)
        r_array = np.array(r_lists)
        if sim_num == 229:
            mark = "o"
        if sim_num == 242:
            mark = "D"
        if sim_num == 329:
            mark = "*"
        for i in range(len(r_array)):
            ax1.scatter(infall_time_array[i], r_array[i], marker = mark, color = color_list[i])
            ax2.scatter(infall_mass_array[i], r_array[i], marker = mark, color = color_list[i])
        ax2.set_xscale('log')
        ax1.set_yscale('log')
        ax2.set_yscale('log')
    
    ax1.set_xlabel("Infall time (Gyr)", fontsize=14)
    ax2.set_xlabel(r"Infall Mass ($M_{\odot}$)", fontsize=14)
        
    ax1.set_ylabel("Radial position of progenitor (kpc)", fontsize=12)
    ax2.set_ylabel("Radial position of progenitor (kpc)", fontsize=12)



    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Elena', markerfacecolor='gray', markersize=12),
        Line2D([0], [0], marker='D', color='w', label='Ruth', markerfacecolor='gray', markersize=12),
        Line2D([0], [0], marker='*', color='w', label='Sonia', markerfacecolor='gray', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Surviving', markerfacecolor='blue', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Disrupted', markerfacecolor='red', markersize=12)
    ]

    
    ax2.legend(handles=legend_elements, loc='best', fontsize=12)

    
    filename = "Analysis2_infall_property_vs_R.svg"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path,
            format='svg',
            dpi=300,
            bbox_inches='tight',
            transparent=False)
    plt.show()
    print(f"--> Plot saved successfully to: {output_path}")

    print("\n--- All snapshots processed. ---")








