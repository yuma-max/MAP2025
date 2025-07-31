#Import Libraries
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


file_path = '/home/takeichi/MAP/Code_Yuma_2025/Code/disrupted_traceCopy_verJul14.py'
module_name = 'stdisrupt'
spec = importlib.util.spec_from_file_location(module_name, file_path)

stdisrupt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stdisrupt)





def plot_com_movement(sim_num):
    """
    This is the function to plot the movement of the COM of the all surviving and disrupted galaxies of the designated simulation

    Args:
        sim_num (int): simulation number which we want to analyze
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
        halo_ids = [hid for hid in halo_ids if not hid.startswith("COM") and not hid.startswith("r")]
        print(f"Successfully found {len(halo_ids)} halos in {base_data_folder}")
    except FileNotFoundError:
        print(f"Error: Directory not found at {base_data_folder}")
        return

    # snapshotlist ---please fix---
    snapshots_to_analyze_str = ['004096', '004032', '003936', '003840', '003744', '003648', '003606', '003552', '003456', '003360', '003264', '003195', '003168', '003072', '002976', '002880', '002784', '002688', '002592', '002554', '002496', '002400', '002304', '002208', '002112', '002088', '002016', '001920', '001824', '001740', '001728', '001632', '001536', '001475', '001440', '001344', '001269', '001248', '001152', '001106', '001056', '000974', '000960', '000864', '000776', '000768', '000672', '000637', '000576', '000480', '000456', '000384', '000347', '000288', '000275', '000225', '000192', '000188', '000139', '000107', '000096', '000071']
    snapshots_to_analyze = [int(s) for s in snapshots_to_analyze_str]
    
    # --- Loading Datasets ---
    try:
        dmainhalo = pd.read_csv(info_data_path)
    except FileNotFoundError:
        print(f"Error: Main halo COM file not found at {info_data_path}")
        return

    # --- Plotting ---
    fig, axes = plt.subplots(2, 3, figsize=(24, 18))
    fig.suptitle(f"Movement of the COM of Galaxies (Sim: {sim_num})", fontsize=20, y=0.95)
    
    # --- loop for each haloids ---
    for item in halo_ids:
        try:
            #get last snapshot and last haloid
            lastsnapshot = int(item[:6])
            halo_id = int(item[7:])
            print(f"Processing data for halo_id: {halo_id} at last_snapshot: {lastsnapshot}")
        except (ValueError, IndexError):
            print(f"Skipping invalid directory name: {item}")
            continue
        file_name=f"{lastsnapshot:06d}{halo_id}"
        print(f"File name should include {file_name}.")
        # Devide snapshot list
        # disrupted_part: snapshots after disrputed
        # surviving_part: snapshots before disrupted
        try:
            split_index = snapshots_to_analyze.index(lastsnapshot)
            disrupted_part = snapshots_to_analyze[:split_index]
            surviving_part = snapshots_to_analyze[split_index:]
        except ValueError:
            print(f"Warning: lastsnapshot {lastsnapshot} for halo {halo_id} not in snapshots_to_analyze list. Skipping.")
            continue

        # Load csv file of each halos
        data_folder = os.path.join(base_data_folder, item)
        csv_path_disrupt = os.path.join(data_folder, f"tracked_star_particles{file_name}.csv")
        csv_path_survive = os.path.join(data_folder, f"halo_star_dist{file_name}.csv")

        df_disrupt, df_survive = None, None
        # there is a possibility that the disrupted part doesn't exist
        if len(disrupted_part)!=0:
            try:
                df_disrupt = pd.read_csv(csv_path_disrupt)
            except FileNotFoundError:
                print(f"Info: Disrupt data file not found for halo {halo_id}, skipping disrupted part.")
                if lastsnapshot != 4096:
                    print("this is problem. Check the files")
        try:
            df_survive = pd.read_csv(csv_path_survive)
        except FileNotFoundError:
            print(f"Error: Survive data file not found for halo {halo_id} at {csv_path_survive}. Skipping this halo.")
            continue

        # Calculate the COM of the halo
        x_com, y_com, z_com, snapshots = [], [], [], []

        # Calculate the COM after disrupted
        if df_disrupt is not None:
            for snapshot in disrupted_part:
                sub_disrupt = df_disrupt[df_disrupt["snapshot"] == snapshot]
                # just in case check that sub_disrupt is not empty
                if not sub_disrupt.empty:
                    mass = sub_disrupt["mass"].to_numpy()
                    M = np.sum(mass)
                    if M > 0:
                        xcom = np.sum(sub_disrupt["x"].to_numpy() * mass) / M
                        ycom = np.sum(sub_disrupt["y"].to_numpy() * mass) / M
                        zcom = np.sum(sub_disrupt["z"].to_numpy() * mass) / M
                        
                      
                        main_halo_com = dmainhalo.loc[dmainhalo['snapshot'] == snapshot, ['0', '1', '2']].values
                        if main_halo_com.size > 0:
                            x1, y1, z1 = main_halo_com[0]
                            x_com.append(xcom - x1)
                            y_com.append(ycom - y1)
                            z_com.append(zcom - z1)
                            snapshots.append(snapshot)

        # Calculate COM before disrupted
        for snapshot in surviving_part:
            sub_survive = df_survive[df_survive["snapshot"] == snapshot]
            if not sub_survive.empty:
                x_com.append(sub_survive["halo_x_com"].iloc[0])
                y_com.append(sub_survive["halo_y_com"].iloc[0])
                z_com.append(sub_survive["halo_z_com"].iloc[0])
                snapshots.append(snapshot)

        if not snapshots:
            print(f"No COM data found for halo {halo_id}. Skipping plot.")
            continue

        # Sort by snapshot
        sorted_lists = sorted(zip(x_com, y_com, z_com, snapshots), key=lambda x: x[3])
        x_com, y_com, z_com, snapshots = [list(t) for t in zip(*sorted_lists)]
        
        # Recalculate the changing point(when it is disrupted)
        try:
            split_plot_index = snapshots.index(lastsnapshot)
        except ValueError:
            continue
            
        # Plot
        is_survivor = (lastsnapshot == 4096)
        row_idx = 0 if is_survivor else 1
        # This automatic classfies each haloid into surviving and disrupted group
        plot_color = "blue" if is_survivor else "red" # color by disruptd or not

        # XY plane
        line, = axes[row_idx, 0].plot(x_com[:split_plot_index+1], y_com[:split_plot_index+1], label=f"Halo {halo_id}", linestyle="-", color=plot_color, alpha=0.7)
        axes[row_idx, 0].plot(x_com[split_plot_index:], y_com[split_plot_index:], linestyle=":", color=line.get_color(), alpha=0.7)
        axes[row_idx, 0].scatter(x_com[0], y_com[0], s=40, marker="s", color=line.get_color())
        axes[row_idx, 0].scatter(x_com[-1], y_com[-1], s=40, marker="*", color=line.get_color())

        # XZ plane
        line, = axes[row_idx, 1].plot(x_com[:split_plot_index+1], z_com[:split_plot_index+1], label=f"Halo {halo_id}", linestyle="-", color=plot_color, alpha=0.7)
        axes[row_idx, 1].plot(x_com[split_plot_index:], z_com[split_plot_index:], linestyle=":", color=line.get_color(), alpha=0.7)
        axes[row_idx, 1].scatter(x_com[0], z_com[0], s=40, marker="s", color=line.get_color())
        axes[row_idx, 1].scatter(x_com[-1], z_com[-1], s=40, marker="*", color=line.get_color())

        # R vs snapshot
        r_com = np.sqrt(np.array(x_com)**2 + np.array(y_com)**2 + np.array(z_com)**2)
        line, = axes[row_idx, 2].plot(snapshots[:split_plot_index+1], r_com[:split_plot_index+1], label=f"Halo {halo_id}", color=plot_color, alpha=0.7)
        axes[row_idx, 2].plot(snapshots[split_plot_index:], r_com[split_plot_index:], color=line.get_color(), linestyle=":", alpha=0.7)

    # --- Setting of Graph ---
    for i in range(2):
        axes[i, 0].set_title("Surviving Galaxies (XY)" if i == 0 else "Disrupted Galaxies (XY)")
        axes[i, 1].set_title("Surviving Galaxies (XZ)" if i == 0 else "Disrupted Galaxies (XZ)")
        axes[i, 2].set_title("Surviving Galaxies (R vs Snap)" if i == 0 else "Disrupted Galaxies (R vs Snap)")
        
        axes[i, 0].set_xlabel("x (kpc)")
        axes[i, 0].set_ylabel("y (kpc)")
        axes[i, 1].set_xlabel("x (kpc)")
        axes[i, 1].set_ylabel("z (kpc)")
        axes[i, 2].set_xlabel("Snapshot Number")
        axes[i, 2].set_ylabel("Distance r (kpc)")
        
        axes[i, 0].grid(True, linestyle='--', alpha=0.6)
        axes[i, 1].grid(True, linestyle='--', alpha=0.6)
        axes[i, 2].grid(True, linestyle='--', alpha=0.6)
        axes[i, 0].legend(loc='best', fontsize='small')


    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Saving Plots
    output_dir = os.path.join(base_data_folder, "COM_motion")
    os.makedirs(output_dir, exist_ok=True)
    filename = f'Movement_COM_sim{sim_num}.png'
    outputpath = os.path.join(output_dir, filename)
    plt.savefig(outputpath)
    print(f"Plot saved to {outputpath}")
    plt.show()




def plot_com_movement_paper1(sim_num_list):
    """
    This is the function to plot the movement of the COM of the all surviving and disrupted galaxies of the designated simulation

    Args:
        sim_num (int): simulation number which we want to analyze
    """
    # --- Plotting ---
    number_of_simnum = len(sim_num_list)

    fig, axes1 = plt.subplots(2, number_of_simnum, figsize=(24, 16))
    fig.suptitle("Movement of the Center of Mass of Satellites over time", fontsize=20, y=0.95)
    
    for i, sim_num in enumerate(sim_num_list):
        if sim_num ==329:
            sim_name ="Elena"
        if sim_num ==229:
            sim_name ="Sonia"
        if sim_num ==242:
            sim_name =="Ruth"
        print(f"Working on {sim_name}")
    # --- Setting the path ---
    # base_data_folder: where the csv file of the motion of the each star particles are saved
    # info_data_path: 
        base_data_folder = f"/home/takeichi/MAP/Results/{sim_num}"
        info_data_path = f"/home/takeichi/MAP/Code_Yuma_2025/Code/Datafiles/main_halo_com_{sim_num}.csv"
    
    # get halo_ids from the datafolder
        try:
            halo_ids = [d for d in os.listdir(base_data_folder) if os.path.isdir(os.path.join(base_data_folder, d))]
            # Exclude directories which start from "COM_motion" or "r".
            halo_ids = [hid for hid in halo_ids if not hid.startswith("COM") and not hid.startswith("r")]
            print(f"Successfully found {len(halo_ids)} halos in {base_data_folder}")
        except FileNotFoundError:
            print(f"Error: Directory not found at {base_data_folder}")
            return

        # snapshotlist ---please fix---
        snapshots_to_analyze_str = ['004096', '004032', '003936', '003840', '003744', '003648', '003606', '003552', '003456', '003360', '003264', '003195', '003168', '003072', '002976', '002880', '002784', '002688', '002592', '002554', '002496', '002400', '002304', '002208', '002112', '002088', '002016', '001920', '001824', '001740', '001728', '001632', '001536', '001475', '001440', '001344', '001269', '001248', '001152', '001106', '001056', '000974', '000960', '000864', '000776', '000768', '000672', '000637', '000576', '000480', '000456', '000384', '000347', '000288', '000275', '000225', '000192', '000188', '000139', '000107', '000096', '000071']
        snapshots_to_analyze = [int(s) for s in snapshots_to_analyze_str]
    
        # --- Loading Datasets ---
        try:
            dmainhalo = pd.read_csv(info_data_path)
        except FileNotFoundError:
            print(f"Error: Main halo COM file not found at {info_data_path}")
            return
        snap_path = f"/data/Sims/h{sim_num}.cosmo50PLK.3072g/h{sim_num}.cosmo50PLK.3072gst5HbwK1BH/snapshots_200crit_h{sim_num}/"
        
    
    # --- loop for each haloids ---
        for item in halo_ids:
            try:
                #get last snapshot and last haloid
                lastsnapshot = int(item[:6])
                halo_id = int(item[7:])
                print(f"Processing data for halo_id: {halo_id} at last_snapshot: {lastsnapshot}")
            except (ValueError, IndexError):
                print(f"Skipping invalid directory name: {item}")
                continue
            file_name=f"{lastsnapshot:06d}{halo_id}"
            print(f"File name should include {file_name}.")
            # Devide snapshot list
            # disrupted_part: snapshots after disrputed
            # surviving_part: snapshots before disrupted
            try:
                split_index = snapshots_to_analyze.index(lastsnapshot)
                disrupted_part = snapshots_to_analyze[:split_index]
                surviving_part = snapshots_to_analyze[split_index:]
            except ValueError:
                print(f"Warning: lastsnapshot {lastsnapshot} for halo {halo_id} not in snapshots_to_analyze list. Skipping.")
                continue

            # Load csv file of each halos
            data_folder = os.path.join(base_data_folder, item)
            csv_path_disrupt = os.path.join(data_folder, f"tracked_star_particles{file_name}.csv")
            csv_path_survive = os.path.join(data_folder, f"halo_star_dist{file_name}.csv")

            df_disrupt, df_survive = None, None
            # there is a possibility that the disrupted part doesn't exist
            if len(disrupted_part)!=0:
                try:
                    df_disrupt = pd.read_csv(csv_path_disrupt)
                except FileNotFoundError:
                    print(f"Info: Disrupt data file not found for halo {halo_id}, skipping disrupted part.")
                    if lastsnapshot != 4096:
                        print("this is problem. Check the files")
            try:
                df_survive = pd.read_csv(csv_path_survive)
            except FileNotFoundError:
                print(f"Error: Survive data file not found for halo {halo_id} at {csv_path_survive}. Skipping this halo.")
                continue

            # Calculate the COM of the halo
            x_com, y_com, z_com, times = [], [], [], []

            # Calculate the COM after disrupted
            if df_disrupt is not None:
                for snapshot in disrupted_part:
                    sub_disrupt = df_disrupt[df_disrupt["snapshot"] == snapshot]
                    # just in case check that sub_disrupt is not empty
                    if not sub_disrupt.empty:
                        mass = sub_disrupt["mass"].to_numpy()
                        M = np.sum(mass)
                        if M > 0:
                            xcom = np.sum(sub_disrupt["x"].to_numpy() * mass) / M
                            ycom = np.sum(sub_disrupt["y"].to_numpy() * mass) / M
                            zcom = np.sum(sub_disrupt["z"].to_numpy() * mass) / M
                        
                      
                            main_halo_com = dmainhalo.loc[dmainhalo['snapshot'] == snapshot, ['0', '1', '2']].values
                            if main_halo_com.size > 0:
                                x1, y1, z1 = main_halo_com[0]
                                x_com.append(xcom - x1)
                                y_com.append(ycom - y1)
                                z_com.append(zcom - z1)
                                halodata=pb.load(f"{snap_path}h{sim_num}.cosmo50PLK.3072gst5HbwK1BH.{snapshot:06d}")
                                #halodata.physical_units()
                                time_at_snap = halodata.properties["time"].in_units("Gyr")
                                times.append(time_at_snap)

            # Calculate COM before disrupted
            for snapshot in surviving_part:
                sub_survive = df_survive[df_survive["snapshot"] == snapshot]
                if not sub_survive.empty:
                    x_com.append(sub_survive["halo_x_com"].iloc[0])
                    y_com.append(sub_survive["halo_y_com"].iloc[0])
                    z_com.append(sub_survive["halo_z_com"].iloc[0])
                    halodata=pb.load(f"{snap_path}h{sim_num}.cosmo50PLK.3072gst5HbwK1BH.{snapshot:06d}")
                    #halodata.physical_units()
                    time_at_snap = halodata.properties["time"].in_units("Gyr")
                    times.append(time_at_snap)

            if not times:
                print(f"No COM data found for halo {halo_id}. Skipping plot.")
                continue

            # Sort by snapshot
            sorted_lists = sorted(zip(x_com, y_com, z_com, times), key=lambda x: x[3])
            x_com, y_com, z_com, times = [list(t) for t in zip(*sorted_lists)]
        
            # Recalculate the changing point(when it is disrupted)
            try:

                halodata=pb.load(f"{snap_path}h{sim_num}.cosmo50PLK.3072gst5HbwK1BH.{lastsnapshot:06d}")
                #halodata.physical_units()
                lasttime = halodata.properties["time"].in_units("Gyr")
                split_plot_index = times.index(lasttime)
            except ValueError:
                continue
            
            # Plot
            is_survivor = (lastsnapshot == 4096)
            column_idx = 0 if is_survivor else 1
            # This automatic classfies each haloid into surviving and disrupted group
            plot_color = "blue" if is_survivor else "red" # color by disruptd or not

            # R vs snapshot
            r_com = np.sqrt(np.array(x_com)**2 + np.array(y_com)**2 + np.array(z_com)**2)
            line, = axes1[column_idx,i].plot(times[:split_plot_index+1], r_com[:split_plot_index+1], color=plot_color, alpha=0.7)
            axes1[column_idx,i].plot(times[split_plot_index:], r_com[split_plot_index:], color=line.get_color(), linestyle=":", alpha=0.7)

        # --- Setting of Graph ---
        for l in range(2):
            axes1[l,i].set_title(f"Surviving Galaxies (R vs Time), {sim_name}" if l == 0 else f"Disrupted Galaxies (R vs Time), {sim_name}")
        
            axes1[l,i].set_xlabel("Time (Gyr)")
            axes1[l,i].set_ylabel("Distance from host (kpc)")
            axes1[l,i].grid(True, linestyle='--', alpha=0.6)

            legend_elements_suv = [
                Line2D([0], [0], color='blue', label='Surviving', lw=2),
            ]
            legend_elements_dis = [
                Line2D([0], [0], color='red', label='Disrupted (before disruption)', lw=2),
                Line2D([0], [0], color='red', label='Disrupted (after disruption)', linestyle=':'),
            ]
            if l==0:
                axes1[l,i].legend(handles=legend_elements_suv, loc='upper right', fontsize=12)
            else:
                axes1[l,i].legend(handles=legend_elements_dis, loc='upper right', fontsize=12)
    
    # Saving Plots
    out_data_folder = "/home/takeichi/MAP/Results"
    output_dir = os.path.join(out_data_folder, "COM_motion")
    os.makedirs(output_dir, exist_ok=True)
    filename = f'Movement.svg'
    outputpath = os.path.join(output_dir, filename)
    plt.savefig(outputpath,
            format='svg',
            dpi=300,
            bbox_inches='tight',
            transparent=False)
    print(f"Plot saved to {outputpath}")
    plt.show()



def plot_com_movement_paper2(sim_num_list):
    """
    This is the function to plot the movement of the COM of the all surviving and disrupted galaxies of the designated simulation

    Args:
        sim_num (int): simulation number which we want to analyze
    """
    # --- Plotting ---
    number_of_simnum = len(sim_num_list)

    fig, axes1 = plt.subplots(2, number_of_simnum, figsize=(24, 16))
    fig.suptitle("Movement of the Center of Mass of Satellites over time", fontsize=20, y=0.95)
    
    for i, sim_num in enumerate(sim_num_list):
        if sim_num ==329:
            sim_name ="Elena"
        elif sim_num ==220:
            sim_name ="Sonia"
        elif sim_num ==242:
            sim_name =="Ruth"
        print(f"Working on {sim_name}")
    # --- Setting the path ---
    # base_data_folder: where the csv file of the motion of the each star particles are saved
    # info_data_path: 
        base_data_folder = f"/home/takeichi/MAP/Results/{sim_num}"
        info_data_path = f"/home/takeichi/MAP/Code_Yuma_2025/Code/Datafiles/main_halo_com_{sim_num}.csv"
    
    # get halo_ids from the datafolder
        try:
            halo_ids = [d for d in os.listdir(base_data_folder) if os.path.isdir(os.path.join(base_data_folder, d))]
            # Exclude directories which start from "COM_motion" or "r".
            halo_ids = [hid for hid in halo_ids if not hid.startswith("COM") and not hid.startswith("r")]
            print(f"Successfully found {len(halo_ids)} halos in {base_data_folder}")
        except FileNotFoundError:
            print(f"Error: Directory not found at {base_data_folder}")
            return

        # snapshotlist ---please fix---
        snapshots_to_analyze_str = ['004096', '004032', '003936', '003840', '003744', '003648', '003606', '003552', '003456', '003360', '003264', '003195', '003168', '003072', '002976', '002880', '002784', '002688', '002592', '002554', '002496', '002400', '002304', '002208', '002112', '002088', '002016', '001920', '001824', '001740', '001728', '001632', '001536', '001475', '001440', '001344', '001269', '001248', '001152', '001106', '001056', '000974', '000960', '000864', '000776', '000768', '000672', '000637', '000576', '000480', '000456', '000384', '000347', '000288', '000275', '000225', '000192', '000188', '000139', '000107', '000096', '000071']
        snapshots_to_analyze = [int(s) for s in snapshots_to_analyze_str]
    
        # --- Loading Datasets ---
        try:
            dmainhalo = pd.read_csv(info_data_path)
        except FileNotFoundError:
            print(f"Error: Main halo COM file not found at {info_data_path}")
            return
        snap_path = f"/data/Sims/h{sim_num}.cosmo50PLK.3072g/h{sim_num}.cosmo50PLK.3072gst5HbwK1BH/snapshots_200crit_h{sim_num}/"
        
    
    # --- loop for each haloids ---
        for item in halo_ids:
            try:
                #get last snapshot and last haloid
                lastsnapshot = int(item[:6])
                halo_id = int(item[7:])
                print(f"Processing data for halo_id: {halo_id} at last_snapshot: {lastsnapshot}")
            except (ValueError, IndexError):
                print(f"Skipping invalid directory name: {item}")
                continue
            file_name=f"{lastsnapshot:06d}{halo_id}"
            print(f"File name should include {file_name}.")
            # Devide snapshot list
            # disrupted_part: snapshots after disrputed
            # surviving_part: snapshots before disrupted
            try:
                split_index = snapshots_to_analyze.index(lastsnapshot)
                disrupted_part = snapshots_to_analyze[:split_index]
                surviving_part = snapshots_to_analyze[split_index:]
            except ValueError:
                print(f"Warning: lastsnapshot {lastsnapshot} for halo {halo_id} not in snapshots_to_analyze list. Skipping.")
                continue

            # Load csv file of each halos
            data_folder = os.path.join(base_data_folder, item)
            csv_path_disrupt = os.path.join(data_folder, f"tracked_star_particles{file_name}.csv")
            csv_path_survive = os.path.join(data_folder, f"halo_star_dist{file_name}.csv")

            df_disrupt, df_survive = None, None
            # there is a possibility that the disrupted part doesn't exist
            if len(disrupted_part)!=0:
                try:
                    df_disrupt = pd.read_csv(csv_path_disrupt)
                except FileNotFoundError:
                    print(f"Info: Disrupt data file not found for halo {halo_id}, skipping disrupted part.")
                    if lastsnapshot != 4096:
                        print("this is problem. Check the files")
            try:
                df_survive = pd.read_csv(csv_path_survive)
            except FileNotFoundError:
                print(f"Error: Survive data file not found for halo {halo_id} at {csv_path_survive}. Skipping this halo.")
                continue

            # Calculate the COM of the halo
            x_com, y_com, z_com, times = [], [], [], []

            # Calculate the COM after disrupted
            if df_disrupt is not None:
                for snapshot in disrupted_part:
                    sub_disrupt = df_disrupt[df_disrupt["snapshot"] == snapshot]
                    # just in case check that sub_disrupt is not empty
                    if not sub_disrupt.empty:
                        mass = sub_disrupt["mass"].to_numpy()
                        M = np.sum(mass)
                        if M > 0:
                            xcom = np.sum(sub_disrupt["x"].to_numpy() * mass) / M
                            ycom = np.sum(sub_disrupt["y"].to_numpy() * mass) / M
                            zcom = np.sum(sub_disrupt["z"].to_numpy() * mass) / M
                        
                      
                            main_halo_com = dmainhalo.loc[dmainhalo['snapshot'] == snapshot, ['0', '1', '2']].values
                            if main_halo_com.size > 0:
                                x1, y1, z1 = main_halo_com[0]
                                x_com.append(xcom - x1)
                                y_com.append(ycom - y1)
                                z_com.append(zcom - z1)
                                halodata=pb.load(f"{snap_path}h{sim_num}.cosmo50PLK.3072gst5HbwK1BH.{snapshot:06d}")
                                #halodata.physical_units()
                                time_at_snap = halodata.properties["time"].in_units("Gyr")
                                times.append(time_at_snap)

            # Calculate COM before disrupted
            for snapshot in surviving_part:
                sub_survive = df_survive[df_survive["snapshot"] == snapshot]
                if not sub_survive.empty:
                    x_com.append(sub_survive["halo_x_com"].iloc[0])
                    y_com.append(sub_survive["halo_y_com"].iloc[0])
                    z_com.append(sub_survive["halo_z_com"].iloc[0])
                    halodata=pb.load(f"{snap_path}h{sim_num}.cosmo50PLK.3072gst5HbwK1BH.{snapshot:06d}")
                    #halodata.physical_units()
                    time_at_snap = halodata.properties["time"].in_units("Gyr")
                    times.append(time_at_snap)

            if not times:
                print(f"No COM data found for halo {halo_id}. Skipping plot.")
                continue

            # Sort by snapshot
            sorted_lists = sorted(zip(x_com, y_com, z_com, times), key=lambda x: x[3])
            x_com, y_com, z_com, times = [list(t) for t in zip(*sorted_lists)]
        
            # Recalculate the changing point(when it is disrupted)
            try:

                halodata=pb.load(f"{snap_path}h{sim_num}.cosmo50PLK.3072gst5HbwK1BH.{lastsnapshot:06d}")
                #halodata.physical_units()
                lasttime = halodata.properties["time"].in_units("Gyr")
                split_plot_index = times.index(lasttime)
            except ValueError:
                continue
            
            # Plot
            is_survivor = (lastsnapshot == 4096)
            #column_idx = 0 if is_survivor else 1
            # This automatic classfies each haloid into surviving and disrupted group
            plot_color = "blue" if is_survivor else "red" # color by disruptd or not

            # R vs snapshot
            line, = axes1[0,i].plot(x_com[:split_plot_index+1], y_com[:split_plot_index+1], color=plot_color, alpha=0.7)
            axes1[0,i].plot(x_com[split_plot_index:], y_com[split_plot_index:], color=line.get_color(), linestyle=":", alpha=0.7)
            axes1[0, i].scatter(x_com[0], y_com[0], s=40, marker="s", color=line.get_color())
            axes1[0, i].scatter(x_com[-1], y_com[-1], s=40, marker="*", color=line.get_color())
            line, = axes1[1,i].plot(x_com[:split_plot_index+1], z_com[:split_plot_index+1], color=plot_color, alpha=0.7)
            axes1[1,i].plot(x_com[split_plot_index:], z_com[split_plot_index:], color=line.get_color(), linestyle=":", alpha=0.7)
            axes1[1, i].scatter(x_com[0], z_com[0], s=40, marker="s", color=line.get_color())
            axes1[1, i].scatter(x_com[-1], z_com[-1], s=40, marker="*", color=line.get_color())

            
        # --- Setting of Graph ---
        for l in range(2):
            axes1[l,i].set_title(f"xy plane {sim_name}" if l == 0 else f"xz plane {sim_name}")
        
            axes1[l,i].set_xlabel("x (kpc)")
            axes1[l,i].set_ylabel("y (kpc)" if l == 0 else "z (kpc)")
            axes1[l,i].grid(True, linestyle='--', alpha=0.6)

            legend_elements= [
                Line2D([0], [0], color='blue', label='Surviving', lw=2),
                Line2D([0], [0], color='red', label='Disrupted (before disruption)', lw=2),
                Line2D([0], [0], color='red', label='Disrupted (after disruption)', linestyle=':'),
                Line2D([0], [0], marker='s', color='w', label='Starting Position', markerfacecolor='gray', markersize=12),
                Line2D([0], [0], marker='*', color='w', label='Final Position', markerfacecolor='gray', markersize=12),
            ]
            axes1[l,i].legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Saving Plots
    out_data_folder = "/home/takeichi/MAP/Results"
    output_dir = os.path.join(out_data_folder, "COM_motion")
    os.makedirs(output_dir, exist_ok=True)
    filename = f'Movement_COM_sim.svg'
    outputpath = os.path.join(output_dir, filename)
    plt.savefig(outputpath,
            format='svg',
            dpi=300,
            bbox_inches='tight',
            transparent=False)
    print(f"Plot saved to {outputpath}")
    plt.show()


