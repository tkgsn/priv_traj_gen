from fmm import FastMapMatch,Network,NetworkGraph,UBODTGenAlgorithm,UBODT,FastMapMatchConfig
from fmm import GPSConfig,ResultConfig
import sys
import os


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
    # fmm_model = FastMapMatch(network,graph,ubodt)


    # input_config = GPSConfig()
    # input_config.file = os.path.join(data_dir, "trips.csv")
    # input_config.id = "id"

    # result_config = ResultConfig()
    # result_config.file = os.path.join(data_dir, "mr.txt")
    # result_config.output_config.write_opath = True
    # print(result_config.to_string())

    # k = 4
    # radius = 0.4
    # gps_error = 0.5
    # fmm_config = FastMapMatchConfig(k,radius,gps_error)

    # status = fmm_model.match_gps_file(input_config, result_config, fmm_config)
    # print(status)

if __name__ == "__main__":
    data_dir = sys.argv[1]
    run(data_dir)