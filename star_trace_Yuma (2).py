import pynbody as pb
import numpy as np
import pandas as pd
import glob
import os
import h5py

# Function to make a math to the directory

# snap: id for snapshot
# sim_base: base directory
# snap and sim_base should be string
# file_base: the explanation of the file???

#the result is simbasefilebase.snap

def snapshot_dir(
    snap: str,
    sim_base: str,
):
    sim = (
        sim_base
        +file_base+f".{snap}"
        #+file_base+f".{snap}"
    )
    return sim


def main(
    halos_grpNow=np.array(
        [674, 9799, 3383, 2026, 1467, 1179, 765, 478, 459, 331, 278, 266, 179, 68, 32]
    ),
    sim_base="/data/REPOSITORY/e12Gals/h148.cosmo50PLK.6144g3HbwK1BH/",
    trace_file="h148.cosmo50PLK.6144g3HbwK1BH.004096/h148.cosmo50PLK.6144g3HbwK1BH.004096.trace_back.hdf5",
    data_folder="../data",
    starting_snapshot=9,
    restart: bool = True,
):
    #Split the sim_base by / and get the third element from the back
    file_base = sim_base.split("/")[-3]
    
    # first load all data files that might be necessary
    trace = pd.read_hdf(trace_file, index_col=0)
    #print("--- Inside module.main: DataFrame just loaded ---")
    #print(trace.head())
    #print(f"Does index contain 8?: {8 in trace.index}")
    #mstar_trace = pd.read_csv(f"{data_folder}/mstar_trace.csv", index_col=0)
    #trace = trace.reset_index()
    #print("--- Columns after reset_index ---")
    #print(trace.columns) 
    #print(trace.head()) 
    #id_column_name = trace.columns[0]
    
    #trace = trace.set_index(id_column_name)
    
    # The column names are the snapshots i have to iterate over
    #cols = np.array(mstar_trace.columns)[::-1]  # reverse order

    # get the list of the index and columns
    cols = [trace.index.name]+trace.columns.to_list()  # CC Add the first snapshot in

    #inverse the list
    cols = cols[::-1]
    # cols = trace.columns.to_list()
    
    # for each halo make a folder if one doesnt exist already
    # {i:05d} change the format of i
    # i.e. 674 -> 00674
    print(f"{halos_grpNow=}")

    #print("--- Inside module.main: Just before the loop ---")
    #print(trace.head())
    #print(f"Does index contain 8?: {8 in trace.index}")

    for i in halos_grpNow:
        os.makedirs(f"{data_folder}/star_trace/{i:05d}", exist_ok=True)

    # its possible the run crashed midway through for some reason.
    # If so, i can restart from the last snapshot
    #start = 9 #0
    #in general, this is for the first loop
    if restart:

        # look in directory for file with star_poistions{snapshot}.npz
        # if it exists, get snapshot number from file name
        # if not, start from earliest snapshot
        #print("restarted")
        print("looking to see if files exist for these halos")
        #get all the files
        files = glob.glob(
            f"{data_folder}/star_trace/{halos_grpNow[-1]:05d}/particle_data.h5"
        )
        #if there is a file
        if len(files) > 0:
            print("files have been found")
            print(f"{files=}")
            # get last snapshot from file name
            with h5py.File(f"{np.sort(files)[-1]}", "a") as f:
                if len(f.keys()) == 0:
                    print("The HDF5 file is empty.")
                else:
                    print("The HDF5 file contains keys/groups:", list(f.keys()))
                # Assuming `iteration` tracks your current simulation step
                print(f"{f.keys()=}")
                last_iteration_key = sorted(f.keys())[-1]
            print(f"{last_iteration_key=}")
            start = np.where(cols == last_iteration_key)[0][0]
            snapshots = cols[start:]
            print(f"snapshots left:{snapshots}")


        # when len(files) = 0:
        else:
            print(f"no files found in {data_folder}/star_trace/{halos_grpNow[0]:05d}/")
            start = starting_snapshot # Manually changed to first snapshot where the halo exists
            snapshots = cols[start:]
            # asume files for these snapshots are empty
            print(f"{snapshots[0]=}")
            s = pb.load(
                sim_base
                +file_base+f".{snapshots[0]}"
                #+ f"file_base.{snapshots[0]}"
            )
            h = s.halos(halo_numbers='v1')  # updated for Pynbody 2 backwards compatability
            for i in halos_grpNow:
                #halo_id_at_snapshot = trace.loc[i, snapshots[0]]

                #if halo_id_at_snapshot == -1 or np.isnan(halo_id_at_snapshot):
                  #  print(f"Skipping snapshot {snapshots[0]} for halo {i} because it is lost (halo_id: {halo_id_at_snapshot})")
                 #   continue 
                #halo_index = halo_id_at_snapshot
                if snapshots[0] in trace.columns:
                    halo_index = trace.loc[str(i), snapshots[0]]
                else:
                    halo_index = i


    
                #halo_index = i  # CC Add the first snapshot in
                if np.isnan(halo_index):
                    print(f"halo {i} does not exist in snapshot {snapshots[0]}")
                    print("moving on to next halo")
                    print(f"halo_index: {halo_index}")
                    print("...")
                    continue
                else:
                    try:
                        #halo_index = trace.loc[str(i), snapshots[0]]
                        if snapshots[0] in trace.columns:
                            halo_index = trace.loc[str(i), snapshots[0]]
                        else:
                            halo_index = i
                        #halo_index = i  # CC Add the first snapshot in
                        print(f"working on halo:{i}")
                        print(f"halo index at current snapshot: {halo_index}")
                        h_i = h.load_copy(halo_index)
                        h_i.physical_units()
                        h_i["iord"]

                        x_i = h_i.star["x"]
                        y_i = h_i.star["y"]
                        z_i = h_i.star["z"]
                        vx_i = h_i.star["vx"]
                        vy_i = h_i.star["vy"]
                        vz_i = h_i.star["vz"]
                        mass_i = h_i.star["mass"]
                        iord_i = h_i.star["iord"]
                        bound_stars = h_i.star["iord"]
                        try:
                            feh_i = h_i.star["feh"]
                        except Exception as e:
                            #print the content of t/home/takeichi/MAP/Datafiles/stellarhalo_trace5/star_trace/00115/particle_data.h5'he error
                            print(f"error: {e}")
                            print(f"there are probably no stars: {len(x_i)=}")
                            feh_i = np.zeros(len(x_i))
                        tform_i = h_i.star["tform"].in_units("Gyr")
                        center = pb.analysis.halo.center(
                            h_i, mode="com", retcen=True, vel=True
                        )

                        #create h5 file
                        with h5py.File(
                            f"{data_folder}/star_trace/{i:05d}/particle_data.h5", "a") as f:
                            # Assuming `iteration` tracks your current simulation step
                           
 # Creating datasets for this iteration
                            f.create_dataset(
                                f"{snapshots[0]}/x", data=x_i, compression="gzip"
                            )
                            f.create_dataset(
                                f"{snapshots[0]}/y", data=y_i, compression="gzip"
                            )
                            f.create_dataset(
                                f"{snapshots[0]}/z", data=z_i, compression="gzip"
                            )
                            f.create_dataset(
                                f"{snapshots[0]}/bound_stars",
                                data=bound_stars,
                                compression="gzip",
                            )
                            f.create_dataset(
                                f"{snapshots[0]}/vx",
                                data=vx_i,
                                compression="gzip",
                            )
                            f.create_dataset(
                                f"{snapshots[0]}/vy",
                                data=vy_i,
                                compression="gzip",
                            )
                            f.create_dataset(
                                f"{snapshots[0]}/vz",
                                data=vz_i,
                                compression="gzip",
                            )
                            f.create_dataset(
                                f"{snapshots[0]}/mass",
                                data=mass_i,
                                compression="gzip",
                            )
                            f.create_dataset(
                                f"{snapshots[0]}/iord",
                                data=iord_i,
                                compression="gzip",
                            )
                            f.create_dataset(
                                f"{snapshots[0]}/feh",
                                data=feh_i,
                                compression="gzip",
                            )
                            f.create_dataset(
                                f"{snapshots[0]}/tform",
                                data=tform_i,
                                compression="gzip",
                            )

                        np.savez(
                            f"{data_folder}/star_trace/{i:05d}/centers.npz",
                            centers=[center],
                        )
                        print("data_sucessfully_saved")
                        print(f"wrote {len(h_i.star)} stars")
                    except Exception as e:
                        print(f"error: {e}")
                        print(
                            "maybe this halo doesnt exit at this snapshot, in this case we'll save empty arrays"
                        )

    # Now all folderes are created. If this is a run after its crashed then snapshots should be set to the last snapshot to now
    # c is the number which shows the position of snap in snapshots
    # snap is a number in snapshits
    # for example, when c=0, snap=snapshots[0]
    #start from snapshots[1]
    #track the data from all snapshots
    for c, snap in enumerate(snapshots[:-1]):
        # hopefully loading the snapshot once will be faster
        print(f"loading snapshot: {snapshots[c+1]}")

        s = pb.load(
            sim_base
            +file_base+f".{snapshots[c+1]}/"
            #+file_base+f".{snapshots[c+1]}"
        )
        h = s.halos(halo_numbers='v1')  # updated for Pynbody 2 backwards compatability
        # now iterate through each halo
        for j in halos_grpNow:
            halo_indices = trace.loc[str(j)].to_numpy()
            halo_indices = np.append(j, halo_indices)[::-1]
            print(f"{halo_indices[start+c+0]=}")
            print(f"{halo_indices[start+c+1]=}")
            try:
                #print(s)
                #print(h)
                #print(j)
                track_stars(
                    s,  # sim
                    h,  # halos
                    j,  # halo's grp at t=13.8 Gyr
                    halo_indices[c+start+1],  # index of halo at this snapshot
                    data_folder,
                    snapshots[c],
                    snapshots[c+1],
                )
            except Exception as e:
                print(f"error: {e}")
                print("error in track_stars in the loop")
                #if os.path.exists(f"{data_folder}/star_trace/{halos_grpNow[-1]:05d}")== True:
                    #print("path exists")
                print(f"halo {j} does not exist in snapshot {snapshots[c+1]}")
                print("moving on to next halo")
                continue

    # for i in snapshot_z_t.index:

#function to 
def track_stars(
    s,
    h,
    halo,  # halo's grp at t=13.8 Gyr
    halo_index,  # index of halo at this snapshot
    data_folder,
    old_snapshot, #start_snapshot0115/particle_data.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)
    current_snapshot, #last_snapshot
):

    #if os.path.exists(f"{data_folder}/star_trace/{halo:05d}")== True:
                    #print("path exists")
    print(f"starting snapshot: {old_snapshot}")
    #print(f"{data_folder=}")
    print(f"ending snapshot: {current_snapshot}")
    print(f"{halo=}")
    
    trace_file = f"{data_folder}/star_trace/{halo:05d}/particle_data.h5"
    #get the iord of the stars which is preserved in the last snapshot
    with h5py.File(trace_file, "r") as f:
        # Assuming `iteration` tracks your current simulation step
        # if theres only one key
        print(f"{f.keys()=}")
        #get the latest key
        last_iteration_key = sorted(f.keys())[-1]
        print(f"{last_iteration_key=} i.e particle data from last snapshot saved")
        last_iteration_group = f[last_iteration_key]
        last_iord = last_iteration_group["iord"][:]

    f = pb.new(star=len(last_iord))
    print(f"{len(last_iord)=}")
    f["iord"] = last_iord

    centers = np.load(f"{data_folder}/star_trace/{halo:05d}/centers.npz")

    # load halo at current time
    print(f"halo_index: {halo_index}")
    h_i = h.load_copy(halo_index)
    h_i.physical_units()  #! always make sure its in physical units
    h_i["iord"]
    print(f"{len(h_i.s['iord'])=}")
    bound_stars = h_i.s["iord"]  # iord of bound stars at this time
    # new stars could have formed in the mean time
    bound_new = np.in1d(bound_stars, last_iord, invert=True)
    indices_bound_new = np.where(bound_new)[0]

    h_ix = h_i.s["x"][indices_bound_new]
    h_iy = h_i.s["y"][indices_bound_new]
    h_iz = h_i.s["z"][indices_bound_new]
    h_ivx = h_i.s["vx"][indices_bound_new]
    h_ivy = h_i.s["vy"][indices_bound_new]
    h_ivz = h_i.s["vz"][indices_bound_new]
    h_imass = h_i.s["mass"][indices_bound_new]
    h_iord = h_i.s["iord"][indices_bound_new]
    h_itform = h_i.s["tform"][indices_bound_new]
    try:
        h_ifeh = h_i.star["feh"]
    except Exception as e:
        print(f"error: {e}")
        print("error in trace function")
        print("there are probably no stars")
        h_ifeh = np.zeros(len(h_ix))
    # these are the stars that are bound now but were not bound in the last snapshot/did not exist in the last snapshot
    # we'll need to keep track of these stars as well
    # they could also be empty arrays -- uhhh hopefully that doesnt crash this

    b = pb.bridge.OrderBridge(f, s, monotonic=False)
    print("bridge succesfully made")
    current_halo_stars = b(f)
    current_halo_stars.physical_units()
    x_i, y_i, z_i, iord_i = (
        current_halo_stars["x"],
        current_halo_stars["y"],
        current_halo_stars["z"],
        current_halo_stars["iord"],
    )
    vx_i, vy_i, vz_i = (
        current_halo_stars["vx"],
        current_halo_stars["vy"],
        current_halo_stars["vz"],
    )
    mass_i = current_halo_stars["mass"]
    print(f"{len(x_i)=}")
    print(f"{len(h_ix)=}")
    try:
        feh_i = current_halo_stars.s["feh"]
    except Exception as e:
        print(f"error: {e}")
        print("error2 in trace function")
        feh_i = np.zeros(len(x_i))
    tform_i = current_halo_stars.s["tform"].in_units("Gyr")

    # append x,y,z,iord,mass_i and h_ix, h_iy, h_iz, h_iord,h_imass to last_save
    # append all three arrays to a new array
    xnew = np.concatenate((x_i, h_ix))
    ynew = np.concatenate((y_i, h_iy))
    znew = np.concatenate((z_i, h_iz))
    iord = np.concatenate((iord_i, h_iord))
    vxnew = np.concatenate((vx_i, h_ivx))
    vynew = np.concatenate((vy_i, h_ivy))
    vznew = np.concatenate((vz_i, h_ivz))
    mass_new = np.concatenate((mass_i, h_imass))
    print(f"stars being saved for halo {halo_index} this snapshot: {len(xnew)}")
    try:
        feh_i = np.concatenate((feh_i, h_ifeh))
    except Exception as e:
        print(f"error: {e}")
        print("error3 in trace function")
        feh_i = np.zeros(len(xnew))
    tform_i = np.concatenate((tform_i, h_itform))

    center_com = pb.analysis.halo.center(h_i, mode="com", retcen=True, vel=True)
    center_ssc = pb.analysis.halo.center(h_i, mode="ssc", retcen=True, vel=True)
    center_hyb = pb.analysis.halo.center(h_i, mode="hyb", retcen=True, vel=True)

    np.savez(
        f"{data_folder}/star_trace/{halo:05d}/centers.npz",
        centers=np.append(centers["centers"], [center_com], axis=0),
    )
    with h5py.File(trace_file, "a") as f:
        # Assuming `iteration` tracks your current simulation step
        # Creating datasets for this iteration
        f.create_dataset(f"{current_snapshot}/x", data=xnew, compression="gzip")
        f.create_dataset(f"{current_snapshot}/y", data=ynew, compression="gzip")
        f.create_dataset(f"{current_snapshot}/z", data=znew, compression="gzip")
        f.create_dataset(f"{current_snapshot}/vx", data=vxnew, compression="gzip")
        f.create_dataset(f"{current_snapshot}/vy", data=vynew, compression="gzip")
        f.create_dataset(f"{current_snapshot}/vz", data=vznew, compression="gzip")
        f.create_dataset(f"{current_snapshot}/mass", data=mass_new, compression="gzip")
        f.create_dataset(f"{current_snapshot}/iord", data=iord, compression="gzip")
        f.create_dataset(f"{current_snapshot}/feh", data=feh_i, compression="gzip")
        f.create_dataset(f"{current_snapshot}/tform", data=tform_i, compression="gzip")
        f.create_dataset(
            f"{current_snapshot}/bound", data=bound_stars, compression="gzip"
        )
        f.create_dataset(
            f"{current_snapshot}/center_com", data=center_com, compression="gzip"
        )
        f.create_dataset(
            f"{current_snapshot}/center_ssc", data=center_ssc, compression="gzip"
        )
        f.create_dataset(
            f"{current_snapshot}/center_hyb", data=center_hyb, compression="gzip"
        )

    print("data_sucessfully_saved")
    print("moving on to next halo")
    print("...")


# main()
# main(halos_grpNow=[39, 27, 10, 2, 3, 4, 5, 6, 7, 10, 11, 12, 15, 17])
# main(
#     halos_grpNow=[
#         21,
#         22,
#         23,
#         28,
#         29,
#         34,
#         47,
#         49,
#         51,
#         55,
#         63,
#         74,
#         76,
#         77,
#         92,
#         93,
#         138,
#     ]
# )

# main(halos_grpNow=[33, 35, 194, 204, 6021])
#main(halos_grpNow=[9799])
