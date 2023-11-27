from scipy.cluster.vq import kmeans
from geopy.distance import distance


def clustering(trajs, k):
    """
    Given a list of trajectories (sequence of lat,lon), get a list of centroids.
    and return a dict mapping cluster_id to its centroid.
    """

    appeared_locations = []
    for traj in trajs:
        for lat,lon in traj:
            appeared_locations.append((lat,lon))

    clusters = kmeans(appeared_locations, k)[0]
    return {i:tuple(clusters[i]) for i in range(len(clusters))}

def make_state_to_centroid_id(gps, id_to_centroid):
    state_to_centroid_id = {}

    for i in range(len(gps)):
        gps[i] = tuple(gps[i])

        # assign each state to the closest centroid
        min_dist = float('inf')
        min_cluster = None
        for id, (lat,lon) in id_to_centroid.items():
            dist = distance(gps[i], (lat,lon))
            if dist < min_dist:
                min_dist = dist
                min_cluster = id
        state_to_centroid_id[i] = min_cluster

    return state_to_centroid_id

def state_traj_to_centroid_id_traj(traj, state_to_centroid_id):
    centroid_id_traj = []
    for state in traj:
        centroid_id_traj.append(state_to_centroid_id[state])
    return centroid_id_traj

def run(trajs, gps, k):
    """
    from the state trajs and gps file,
    get the centroid trajs
    """
    latlon_trajs = []
    for traj in trajs:
        latlon_trajs.append([tuple(gps[state]) for state in traj])

    id_to_centroid = clustering(latlon_trajs, k)
    state_to_centroid_id = make_state_to_centroid_id(gps, id_to_centroid)

    centroid_trajs = []
    for traj in trajs:
        centroid_id_traj = state_traj_to_centroid_id_traj(traj, state_to_centroid_id)
        centroid_trajs.append(centroid_id_traj)

    return centroid_trajs, state_to_centroid_id