import config.folder_and_file_names as fname
from discretization.get_discretization import DisData
from primarkov.build_markov_model import ModelBuilder
from generator.state_trajectory_generation import StateGeneration
from generator.to_real_translator import RealLocationTranslator
# from tools.object_store import ObjectStore
from config.parameter_carrier import ParameterCarrier
from config.parameter_setter import ParSetter
from tools.data_writer import DataWriter
from data_preparation.data_preparer import DataPreparer
import datetime
import pandas as pd
import config.folder_and_file_names as config
import pathlib
import json
import pickle

if __name__ == "__main__":
    writer = DataWriter()
    print('begin all')
    print(datetime.datetime.now())
    par = ParSetter().set_up_args()
    pc = ParameterCarrier(par)
    data_preparer = DataPreparer(par)
    trajectory_set = data_preparer.get_trajectory_set(pc)
    print(trajectory_set.trajectory_list[0].trajectory_array.shape)
    disdata1 = DisData(pc)
    grid = disdata1.get_discrete_data(trajectory_set)
    mb1 = ModelBuilder(pc)
    mo1 = mb1.build_model(grid, trajectory_set)
    # mb1 = ModelBuilder(pc)
    mo1 = mb1.filter_model(trajectory_set, grid, mo1)
    sg1 = StateGeneration(pc)
    st_tra_list = sg1.generate_tra(mo1)
    df = pd.DataFrame(st_tra_list)
    df = df.fillna(10000).astype(int)
    
    save_path = pathlib.Path(config.trajectory_data_folder) / "results" / pc.dataset / pc.data_name / pc.training_data_name / pc.save_name
    save_path.mkdir(parents=True, exist_ok=True)
    print("save to", save_path / f"gene.csv")
    df.to_csv(save_path / f"gene.csv", header=None, index=None)
    rlt1 = RealLocationTranslator(pc)
    real_tra_list = rlt1.translate_trajectories(grid, st_tra_list)
    writer.save_trajectory_data_in_list_to_file(real_tra_list, fname.result_file_name)
    print('end all')
    print(datetime.datetime.now())

    print(f"save grid info to", save_path / f"grid_info.pickle")
    grid_info = [grid.level2_borders, grid.level2_x_bin_dict, grid.level2_y_bin_dict]
    print(grid_info)
    with open(save_path / f"grid_info.pickle", "wb") as f:
        pickle.dump(grid_info, f)

    print(vars(pc))
    # save parameters
    with open(save_path / f"params.json", "w") as f:
        json.dump(vars(pc), f)

    # params = {}
    # params['level1_divide'] = grid.level1_x_divide_parameter
    # params['max_num_subdivide'] = grid.level2_x_bin_dict
    # print(params['max_num_subdivide'])
    # with open(pathlib.Path(config.trajectory_data_folder) / "privtrace_params.json", "w") as f:
    #     json.dump(params, f)

    # pass