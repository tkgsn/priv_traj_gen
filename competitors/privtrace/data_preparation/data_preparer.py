from data_preparation.trajectory import Trajectory
from data_preparation.trajectory_set import TrajectorySet
from tools.data_reader import DataReader
from config.parameter_carrier import ParameterCarrier
from config import folder_and_file_names
import pathlib
import numpy as np
from tools.data_writer import DataWriter
import config.folder_and_file_names as config


def is_inside(lat, lon, lat_range, lon_range):
    if lat_range[0] <= lat <= lat_range[1] and lon_range[0] <= lon <= lon_range[1]:
        return True
    else:
        return False

class DataPreparer:

    def __init__(self, args):
        self.cc = ParameterCarrier(args)

    def get_trajectory_set(self, pc):
        tr_set = TrajectorySet()
        reader1 = DataReader()
        load_path = pathlib.Path(folder_and_file_names.trajectory_data_folder) / self.cc.dataset / self.cc.data_name / self.cc.training_data_name / "privtrace_training_data.dat"
        print("load from", load_path)
        raw_tr_list = reader1.read_trajectories_from_data_file(load_path)
        tr_list = []
        print("removing traj that includes points outside the range")
        for traj in raw_tr_list:
            check_insides = [is_inside(lat, lon, pc.lat_range, pc.lon_range) for lat, lon in traj]
            if all(check_insides):
                tr_list.append(traj)
        print(f"the number of traj is {len(raw_tr_list)} -> {len(tr_list)}")

        save_path = pathlib.Path(config.trajectory_data_folder) / "results" / self.cc.dataset / self.cc.data_name /self.cc.training_data_name
        save_path.mkdir(parents=True, exist_ok=True)
        writer = DataWriter()
        writer.save_trajectory_data_in_list_to_file(tr_list, save_path / "training_data.csv")
        print("save training data to", save_path / "training_data.csv")
        for tr_array in tr_list:
            tr = Trajectory()
            tr.trajectory_array = tr_array
            tr_set.add_trajectory(tr)
        return tr_set
