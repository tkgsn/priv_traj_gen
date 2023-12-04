import pickle
import numpy as np
import pathlib
import json

from generator.trajectory_generator import Generator
from config.parameter_carrier import ParameterCarrier
from config.parameter_setter import ParSetter
from data_preparation.data_preparer import DataPreparer
from discretization.get_discretization import DisData
from primarkov.build_markov_model import ModelBuilder
import privtrace_generator

import sys
sys.path.append("../../")
from grid import Grid

class PrivTraceGenerator():

    def __init__(self, cc, markov, privtrace_id_to_states):
        self.privtrace_id_to_states = privtrace_id_to_states
        self.state_to_privtrace_id = {}
        for privtrace_id, states in privtrace_id_to_states.items():
            for state in states:
                self.state_to_privtrace_id[state] = privtrace_id

        generator = Generator(cc)
        generator.load_generator(markov)
        self.privtrace_model = generator
        self.neighbor_check = False

    def eval(self):
        pass

    def train(self):
        pass

    def make_sample(self, references, mini_batch_size):
        '''
        return the mini_batch_size trajectories
        in this implementation, the generated trajectory has the same start state as the reference, but it does not have the same trajectory length
        '''
        sampled = []
        for reference in references:
            start_state = reference[0]
            start_privtrace_id = self.state_to_privtrace_id[start_state]
            # generate_trajectory = self.privtrace_model.generate_trajectory(start_privtrace_id=start_privtrace_id)
            generated_trajectory = None
            counter = 0
            while generated_trajectory is None:
                generated_trajectory = self.privtrace_model.generate_trajectory(neighbor_check=self.neighbor_check, start_privtrace_id=start_privtrace_id)
                if type(generated_trajectory) == bool:
                    generated_trajectory = None
                counter += 1
                if counter > 100:
                    raise Exception("cannot generate trajectory that starts with", start_privtrace_id)

            # the start state of generated trajectory can be different because privtrace_id area is larger than the area of the state
            generated_trajectory = self.post_process(generated_trajectory, start_state)

            assert generated_trajectory[0] == start_state
            sampled.append(generated_trajectory)
        
        return sampled

    def post_process(self, trajectory, start_state):
        new_traj = []
        for privtrace_id in trajectory:
            states = self.privtrace_id_to_states[privtrace_id]
            # randomly choose a state from states
            state = np.random.choice(states)
            new_traj.append(state)
        new_traj[0] = start_state
        return new_traj


def make_privtrace_id_to_states(grid, lat_range, lon_range, n_bins):

    ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
    our_grid = Grid(ranges)

    border_latlons = grid.level2_borders
    for border_latlon in border_latlons:
        lat_max, lat_min, lon_min, lon_max = border_latlon

        # central_points_gps = grid.usable_state_central_points()
        # state_to_center_latlon = central_points_gps

        privtrace_id_to_states = {}
        for i, border in enumerate(border_latlons):
            lat_max, lat_min, lon_min, lon_max = border
            # add padding value to the border to make it a little bit larger
            lat_max += 1e-4
            lat_min -= 1e-4
            lon_min -= 1e-4
            lon_max += 1e-4
            # find the states that are completely in the border
            states = []
            for state, state_border in our_grid.grids.items():
                lon_range, lat_range = state_border
                left_lon, right_lon = lon_range
                bottom_lat, top_lat = lat_range
                # print(left_lon, lon_min, right_lon, lon_max, bottom_lat, lat_min, top_lat, lat_max)
                if left_lon >= lon_min and right_lon <= lon_max and top_lat <= lat_max and bottom_lat >= lat_min:
                    states.append(state)
            # "privtrace 2nd layer cell should include states whose number is power of 2 and >0"
            # print(i, border, len(states), states)
            assert len(states) > 0 and (len(states) & (len(states)-1)) == 0, f"len(states)={len(states)}"
            privtrace_id_to_states[i] = states

    return privtrace_id_to_states


if __name__ == "__main__":
    par = ParSetter().set_up_args()
    pc = ParameterCarrier(par)

    data_preparer = DataPreparer(par)
    trajectory_set = data_preparer.get_trajectory_set(pc)

    disdata1 = DisData(pc)
    grid = disdata1.get_discrete_data(trajectory_set)

    mb1 = ModelBuilder(pc)
    mo1 = mb1.build_model(grid, trajectory_set)
    mo1 = mb1.filter_model(trajectory_set, grid, mo1)

    # same setting needs to be used
    lat_range = par["lat_range"]
    lon_range = par["lon_range"]
    n_bins = par["n_bins"]
    privtrace_id_to_states = privtrace_generator.make_privtrace_id_to_states(grid, lat_range, lon_range, n_bins)
    generator = privtrace_generator.PrivTraceGenerator(pc, mo1, privtrace_id_to_states)

    # save to f"/data/results/{dataset}/{data_name}/{training_data_name}"
    save_path = pathlib.Path("/data") / par["dataset"] / par["data_name"] / par["training_data_name"] / par["save_name"] / f"model_{par['total_epsilon']}"
    save_path.mkdir(parents=True, exist_ok=True)
    print("save to", save_path / f"privtrace_generator.pickle")
    with open(save_path / f"privtrace_generator.pickle", "wb") as f:
        pickle.dump(generator, f)
    print("save to", save_path / f"privtrace_id_to_states.json")
    with open(save_path / f"privtrace_id_to_states.json", "w") as f:
        json.dump(privtrace_id_to_states, f)

    with open(save_path.parent / f"params.json", "w") as f:
        par["network_type"] = "privtrace"
        par["training_data_dir"] = str(save_path.parent.parent)
        json.dump(par, f)