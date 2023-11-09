# requires python2
from fmm import FastMapMatch,Network,NetworkGraph,UBODTGenAlgorithm,UBODT,FastMapMatchConfig
from fmm import GPSConfig,ResultConfig
import sys
import os


def convert_mr_to_training(data_dir, save_dir):
    # format of training: edge_id edge_id ... edge_id 0

    # load times
    with open(os.path.join(data_dir, "times.csv"), "r") as f:
        f.readline()
        times = []
        for line in f:
            time = line.split(",")
            time = [float(t) for t in time if t != ""]
            times.append(time)

    training_data = []
    training_data_time = []
    n_strange = 0
    with open(os.path.join(data_dir, "mr.txt"), "r") as f:
        f.readline()
        for line in f:
            id = int(line.split(";")[0])
            edge_ids_for_each_point = line.split(";")[1]
            edge_ids = line.split(";")[2]
            wkt = line.split(";")[3]

            edge_ids = edge_ids.split(",")
            # convert to int
            edge_ids = [int(edge_id) for edge_id in edge_ids if edge_id != ""]
            # if it includes 0, it means that map matching failed
            if len(edge_ids) == 0:
                continue
            # edge_ids.append(0)

            edge_ids_for_each_point = edge_ids_for_each_point.split(",")
            # convert to int
            edge_ids_for_each_point = [int(edge_id) for edge_id in edge_ids_for_each_point if edge_id != ""]

            assert len(times[id-1]) == len(edge_ids_for_each_point), f"{len(times[id-1])} != {len(edge_ids_for_each_point)}"

            # get the indice that change the edge
            change_edge_indices = [0] + [i+1 for i in range(len(edge_ids_for_each_point)-1) if edge_ids_for_each_point[i] != edge_ids_for_each_point[i+1]]
            edge_ids_ = [edge_ids_for_each_point[i] for i in change_edge_indices] + [0]
            # get the time of the change
            change_times = [times[id-1][i] for i in change_edge_indices]
            # get the difference of the time
            change_times = [0] + [int(change_times[i+1]-change_times[i]) for i in range(len(change_times)-1)]

            # edge_ids_ <- original edges
            # edge_ids <- connected by compensation if two adjacent edges are not connected
            # add 0 to the times where the edge is compensated
            cursor = 0
            for i in range(len(edge_ids_)-1):
                current_edge = edge_ids_[i]
                while current_edge != edge_ids[cursor]:
                    cursor += 1
                    change_times.insert(cursor, 0)
                cursor += 1

            if len(change_times) != len(edge_ids):
                n_strange += 1
                print("WARNING: diffenrt length", len(change_times), len(edge_ids), n_strange)
                print(edge_ids, edge_ids_)
                edge_ids = edge_ids[:len(change_times)]
            # if len(edge_ids_) != len(edge_ids)+1:
                # print("skip because an edge is not connected")
                # continue

            training_data_time.append(change_times)
            training_data.append(edge_ids + [0])
            assert len(change_times) == len(edge_ids), f"{len(change_times)} != {len(edge_ids)}"
    
    with open(os.path.join(save_dir, "training_data.csv"), "w") as f:
        for edge_ids in training_data:
            f.write(" ".join([str(edge_id) for edge_id in edge_ids])+"\n")
    
    with open(os.path.join(save_dir, "training_data_time.csv"), "w") as f:
        for times in training_data_time:
            f.write(" ".join([str(time) for time in times])+"\n")


def run(data_dir):
    network = Network(os.path.join(data_dir, "edges.shp"),"fid", "u", "v")
    print("Nodes {} edges {}".format(network.get_node_count(),network.get_edge_count()))
    graph = NetworkGraph(network)

    # Can be skipped if you already generated an ubodt file
    ubodt_gen = UBODTGenAlgorithm(network,graph)
    # The delta is defined as 3 km approximately. 0.03 degrees. 
    status = ubodt_gen.generate_ubodt(os.path.join(data_dir, "ubodt.txt"), 0.03, binary=False, use_omp=True)
    # Binary is faster for both IO and precomputation
    # status = ubodt_gen.generate_ubodt("stockholm/ubodt.bin", 0.03, binary=True, use_omp=True)
    print(status)

    ubodt = UBODT.read_ubodt_csv(os.path.join(data_dir, "ubodt.txt"))
    ### Create FMM model
    fmm_model = FastMapMatch(network,graph,ubodt)


    input_config = GPSConfig()
    input_config.file = os.path.join(data_dir, "trips.csv")
    input_config.id = "id"

    result_config = ResultConfig()
    result_config.file = os.path.join(data_dir, "mr.txt")
    result_config.output_config.write_opath = True
    print(result_config.to_string())

    k = 4
    radius = 0.4
    gps_error = 0.5
    fmm_config = FastMapMatchConfig(k,radius,gps_error)

    status = fmm_model.match_gps_file(input_config, result_config, fmm_config)
    print(status)

    convert_mr_to_training(data_dir, data_dir)

if __name__ == "__main__":
    data_dir = sys.argv[1]
    run(data_dir)