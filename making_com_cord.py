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


file_path = '/home/takeichi/MAP/Code_Yuma_2025/Code/disrupted_traceCopy_verJul14.py'
module_name = 'stdisrupt'
spec = importlib.util.spec_from_file_location(module_name, file_path)

stdisrupt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stdisrupt)

def make_COM_file(sim_num):
    stdisrupt.mainhaloid_trace(sim_num)
    trace_file = f'/data/Sims/h{sim_num}.cosmo50PLK.3072g/h{sim_num}.cosmo50PLK.3072gst5HbwK1BH/h{sim_num}.cosmo50PLK.3072gst5HbwK1BH.004096/h{sim_num}.cosmo50PLK.3072gst5HbwK1BH.004096.trace_back3_1.hdf5'
    mainhaloid = pd.read_hdf(trace_file, index_col=0)
    print("This is the main halo id trace.")
    mainhaloid

    main_com_cache = {}
    snapshot_list= ['004096', '004032', '003936', '003840', '003744', '003648', '003606', '003552', '003456', '003360', '003264', '003195', '003168', '003072', '002976', '002880', '002784', '002688', '002592', '002554', '002496', '002400', '002304', '002208', '002112', '002088', '002016', '001920', '001824', '001740', '001728', '001632', '001536', '001475', '001440', '001344', '001269', '001248', '001152', '001106', '001056', '000974', '000960', '000864', '000776', '000768', '000672', '000637', '000576', '000480', '000456', '000384', '000347', '000288', '000275', '000225', '000192', '000188', '000139', '000107', '000096', '000071']
    for snap in snapshot_list:
        try:
            xmain, ymain, zmain = stdisrupt.COM_main_halo(sim_num, int(snap), mainhaloid)
            main_com_cache[snap] = (xmain, ymain, zmain)
        except Exception as e:
            print(f"  -> Could not calculate COM for snapshot {snap}. Details: {e}")
            main_com_cache[snap] = (None, None, None) # Mark as failed

    print(f"COM calculation finished.")



    output_folder = "/home/takeichi/MAP/Code_Yuma_2025/Code/Datafiles"
    os.makedirs(output_folder, exist_ok=True) 
    com_df = pd.DataFrame.from_dict(main_com_cache, orient='index')
    com_df.index.name = 'snapshot'
    com_df.sort_index(inplace=True)
    com_output_path = os.path.join(output_folder, f"main_halo_com_{sim_num}.csv")
    try:
        com_df.to_csv(com_output_path)
    except Exception as e:
        print(f"{e}")

