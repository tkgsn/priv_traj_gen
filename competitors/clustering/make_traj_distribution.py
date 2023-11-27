from collections import Counter

import sys
sys.path.append("../../")

from my_utils import add_noise, noise_normalize

def make_id_to_traj(trajs):
    """
    Given a list of trajectories, return a dict mapping id to trajectory.
    """
    trajs = list(set([tuple(traj) for traj in trajs]))
    return {i:traj for i,traj in enumerate(trajs)}


def make_traj_count(traj_list):
    """
    Given a list of trajectories, return a dict mapping trajectory to its frequency.
    """
    return Counter(tuple(traj) for traj in traj_list)

def run(trajs, epsilon):
    """
    from the trajs, computing the distribution of trajs with laplace noise added with epsilon-DP
    """
    id_to_traj = make_id_to_traj(trajs)
    traj_count = make_traj_count(trajs)

    counts = [0]*len(id_to_traj)
    for i in range(len(id_to_traj)):
        counts[i] = traj_count[id_to_traj[i]]

    # add noise
    noisy_traj_count = add_noise(counts, 1, epsilon)
    noisy_traj_distribution = noise_normalize(noisy_traj_count)

    return id_to_traj, noisy_traj_distribution