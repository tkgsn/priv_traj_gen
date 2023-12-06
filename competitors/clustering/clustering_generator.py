import sys
import numpy as np

sys.path.append("../../")
from  my_utils import noise_normalize, add_noise

class ClusteringGenerator():
    '''
    This generator returns the trajectory by the distribution
    '''
    def __init__(self, count, id_to_traj, state_to_centroid_id, epsilon):
        self.n_centroids = len(set(state_to_centroid_id.values()))
        self.noisy_count = add_noise(count, 1, epsilon)
        self.distribution = noise_normalize(self.noisy_count)
        self.outsider_probs = self.compute_outsider_probs(self.noisy_count, self.n_centroids, id_to_traj, epsilon)
        print("outsider_probs", self.outsider_probs)
        self.epsilon = epsilon
        self.id_to_traj = id_to_traj
        self.state_to_centroid_id = state_to_centroid_id
        self.centroid_id_to_states = self.make_centroid_id_to_states(state_to_centroid_id)

    def compute_outsider_probs(self, noisy_count, n_centroids, id_to_traj, epsilon):
        """
        this function solves the computation and memory problem of the huge number of outsiders
        there are huge number of outsiders (the trajectories that are not in the training data i.e., O(n_centroids^seq_len)), so we cannot store and compute distributions of them
        therefore, we first sample whether the generated trajectory is outsider or not, and then sample the outsider trajectory from the uniform distribution (i.e., Pr(outsider) = 1/n_outsiders)
        the probability of outsider is computed as follows:
        Pr(outsider) = sum_outsider |Laplace| / (sum_outsider |Laplace| + sum noisy_counts)
        For each seq_len (from seq_len=2...), we compute the probability of outsider
        If the probability of outsider is over 0.99, we set the probability to the previous probability (i.e., >0.99) to the rest of seq_len
        Therefore, for seq_len larger than the seq_len with 0.99 probability of outsider, a little bit advantage is given to the outsider than the strict DP
        """
        max_seq_len = max([len(traj) for traj in id_to_traj.values()])
        if epsilon == 0:
            return {seq_len:0 for seq_len in range(2, max_seq_len+1)}
        threshold = 0.99
        outsider_probs = {}
        # print(id_to_traj)
        for seq_len in range(2, max_seq_len+1):
            seq_len_ids = [id for id, traj in id_to_traj.items() if len(traj) == seq_len]
            n_seq_len_trajs = len(seq_len_ids)
            n_outsiders = n_centroids**(seq_len) - n_seq_len_trajs
            print(n_outsiders)

            sum_of_outsider_noisy_count = np.abs(np.random.laplace(loc=0, scale=1/epsilon, size=n_outsiders)).sum()
            sum_of_noisy_count = abs(sum([noisy_count[id] for id in seq_len_ids]))
            outsider_prob = sum_of_outsider_noisy_count / (sum_of_outsider_noisy_count + sum_of_noisy_count)
            outsider_probs[seq_len] = outsider_prob
            # print(n_outsiders, sum_of_outsider_noisy_count, sum_of_noisy_count, outsider_prob, seq_len_ids)
            if outsider_prob > threshold:
                break
        
        for i in range(seq_len, max_seq_len+1):
            outsider_probs[i] = outsider_prob

        return outsider_probs
    
    def make_centroid_id_to_states(self, state_to_centroid_id):
        centroid_id_to_states = {}
        for state, centroid_id in state_to_centroid_id.items():
            if centroid_id not in centroid_id_to_states:
                centroid_id_to_states[centroid_id] = []
            centroid_id_to_states[centroid_id].append(state)

        for centroid_id, states in centroid_id_to_states.items():
            if len(states) == 0:
                print("WARNING: centroid_id {} has no state".format(centroid_id))
                centroid_id_to_states[centroid_id] = [0]
        
        return centroid_id_to_states

    def seq_len_to_ids(self, seq_len):
        '''
        return the ids of the trajectories with the same length as seq_len
        '''
        ids = []
        for id, traj in self.id_to_traj.items():
            if len(traj) == seq_len:
                ids.append(id)
        
        return ids

    def reference_to_ids(self, reference):
        '''
        return the ids of the trajectories with the same reference format
        '''
        start_state = reference[0]
        start_centroid_id = self.state_to_centroid_id[start_state]
        seq_len = len(reference)
        ids = []
        for id, traj in self.id_to_traj.items():
            if len(traj) == seq_len and traj[0] == start_centroid_id:
                ids.append(id)
        
        return ids

    def sample_from_ids(self, ids):
        distribution = [self.distribution[id] for id in ids]
        distribution = noise_normalize(distribution)

        return np.random.choice(ids, p=distribution)

    def eval(self):
        pass

    def train(self):
        pass

    def sample_state(self, centroid_id, previous_location=None):
        """
        uniformly randomly sample a location from the cluster
        """
        states = self.centroid_id_to_states[centroid_id]
        # remove previous_location
        if previous_location is not None:
            states = [state for state in states if state != previous_location]

        if len(states) == 0:
            if previous_location > 0:
                return previous_location-1
            else:
                return previous_location+1
        else:
            return np.random.choice(states)

    def post_process(self, traj, reference):
        """
        post process the generated trajectory
        applying sample_location to each cluster_id
        """
        post_processed_traj = [reference[0]]
        for centroid_id in traj[1:]:
            post_processed_traj.append(self.sample_state(centroid_id, post_processed_traj[-1]))
        return post_processed_traj

    def make_sample(self, references, mini_batch_size):
        '''
        return the mini_batch_size trajectories with the same size as the refenreces
        '''
        sampled = []
        for reference in references:
            seq_len = len(reference)
            is_outsider = np.random.binomial(1, self.outsider_probs[seq_len])
            if is_outsider:
                """
                randomly maki212ng the trajectory that is not in the training data
                """
                while True:
                    traj = [reference[0]]
                    for _ in range(seq_len-1):
                        # choose from n_centroids execpt for the first state
                        candidates = list(range(self.n_centroids))
                        # candidates.remove(traj[-1])
                        traj.append(np.random.choice(candidates)) 
                    if tuple(traj) not in self.id_to_traj.values():
                        break
            else:
                # ids = self.seq_len_to_ids(seq_len)
                ids = self.reference_to_ids(reference)
                sampled_id = self.sample_from_ids(ids)
                traj = self.post_process(self.id_to_traj[sampled_id], reference)

            # ids = self.seq_len_to_ids(seq_len)
            # ids = self.reference_to_ids(reference)
            # sampled_id = self.sample_from_ids(ids)
            sampled.append(traj)
        
        return sampled