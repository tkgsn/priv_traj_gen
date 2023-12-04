import sys
import numpy as np

sys.path.append("../../")
from  my_utils import noise_normalize

class ClusteringGenerator():
    '''
    This generator returns the trajectory by the distribution
    '''
    def __init__(self, distribution, id_to_traj, state_to_centroid_id):
        self.distribution = distribution
        self.id_to_traj = id_to_traj
        self.state_to_centroid_id = state_to_centroid_id
        self.centroid_id_to_states = self.make_centroid_id_to_states(state_to_centroid_id)
    
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
            # ids = self.seq_len_to_ids(seq_len)
            ids = self.reference_to_ids(reference)
            sampled_id = self.sample_from_ids(ids)
            traj = self.post_process(self.id_to_traj[sampled_id], reference)
            sampled.append(traj)
        
        return sampled