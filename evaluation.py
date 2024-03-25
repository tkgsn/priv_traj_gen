import torch
import numpy as np
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle

from name_config import make_model_dir, make_training_data_path, make_save_name
from my_utils import construct_default_quadtree, noise_normalize, save, plot_density, get_datadir, set_logger, get_original_dataset_name
from collections import Counter
import numpy as np
import scipy
import random
import pathlib
import sqlite3
import tqdm
import json
import pyemd
import subprocess

import sys
sys.path.append("./competitors/privtrace")
from competitors.privtrace.privtrace_generator import PrivTraceGenerator

sys.path.append("./competitors/clustering")
from competitors.clustering.clustering_generator import ClusteringGenerator


def make_downsampling_dict(from_bin, to_bin):
    assert from_bin > to_bin, "from_bin must be larger than to_bin"

    from grid import Grid
    # lat_range, lon_range do not affect
    lat_range = [30, 40]
    lon_range = [110, 120]
    ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, from_bin)
    from_grid = Grid(ranges)

    ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, to_bin)
    to_grid = Grid(ranges)

    downsample_dict = {}
    for to_state, border in to_grid.grids.items():
        lon_range, lat_range = border
        lon_min, lon_max = lon_range
        lat_min, lat_max = lat_range

        lon_min -= 1e-5
        lon_max += 1e-5
        lat_min -= 1e-5
        lat_max += 1e-5

        # find the states that are completely in the border
        for from_state, state_border in from_grid.grids.items():
            lon_range, lat_range = state_border
            left_lon, right_lon = lon_range
            bottom_lat, top_lat = lat_range

            if left_lon >= lon_min and right_lon <= lon_max and top_lat <= lat_max and bottom_lat >= lat_min:
                downsample_dict[from_state] = to_state
    
    return downsample_dict

def downsample_trajs(trajs, downsampling_dict):
    new_trajs = []
    indice = []
    for id, traj in enumerate(trajs):

        new_traj = [downsampling_dict[traj[0]]]
        for i in range(len(traj)-1):
            downsampled_state = downsampling_dict[traj[i+1]]
            if downsampled_state != new_traj[-1]:
                new_traj.append(downsampled_state)
        # if len(new_traj) > 1:
        new_trajs.append(new_traj)
        indice.append(id)
    return new_trajs, indice

def make_counting_functions(n_base_locations, **kwargs):

    # global: generated -> Counter
    def evaluate_passing(generated_stay_trajs, generated_route_trajs, dataset, counter):
        counter += count_passing_locations(generated_route_trajs)
    def evaluate_source(generated_stay_trajs, generated_route_trajs, dataset, counter):
        counter += count_source_locations(generated_stay_trajs)
    def evaluate_distance(generated_stay_trajs, generated_route_trajs, dataset, counter):
        counter += count_distance(dataset.distance_matrix, generated_stay_trajs, dataset.n_bins_for_distance)
    def first_count(generated_stay_trajs, generated_route_trajs, dataset, counter):
        first_locations = [traj[0] for traj in generated_stay_trajs if len(traj) > 1]
        counter += Counter(first_locations)

    # conditional: generated -> list(Counter)
    def evaluate_emp_next(generated_stay_trajs, generated_route_trajs, dataset, counter):
        for location, counter_ in zip(dataset.top_base_locations, counter):
            counter_ += count_first_next_locations(generated_stay_trajs, location)
    def evaluate_target(generated_stay_trajs, generated_route_trajs, dataset, counter):
        for location, counter_ in zip(dataset.top_base_locations, counter):
            counter_ += count_target_locations(generated_stay_trajs, location)
    def evaluate_destination(generated_stay_trajs, generated_route_trajs, dataset, counter):
        for location, counter_ in zip(dataset.top_base_locations, counter):
            counter_ += compute_destination_count(generated_stay_trajs, location)
    def evaluate_route(generated_stay_trajs, generated_route_trajs, dataset, counter):
        for location, counter_ in zip(dataset.top_base_locations, counter):
            counter_ += count_route_locations(generated_route_trajs, location)

    evaluation_functions = []
    counters = []
    evaluating_metrics = []
    # if kwargs["evaluate_global"]:
        # evaluation_functions.append(compute_global_counts_from_time_label)
    if kwargs["evaluate_passing"]:
        evaluation_functions.append(evaluate_passing)
        counters.append(Counter())
        evaluating_metrics.append("passing")
    if kwargs["evaluate_source"]:
        evaluation_functions.append(evaluate_source)
        counters.append(Counter())
        evaluating_metrics.append("source")
    if kwargs["evaluate_emp_next"]:
        evaluation_functions.append(evaluate_emp_next)
        counters.append([Counter() for _ in range(n_base_locations)])
        evaluating_metrics.append("emp_next")
    # if kwargs["evaluate_second_emp_next"]:
    #     evaluation_functions.append(count_second_order_first_next_locations)
    if kwargs["evaluate_target"]:
        evaluation_functions.append(evaluate_target)
        counters.append([Counter() for _ in range(n_base_locations)])
        evaluating_metrics.append("target")
    if kwargs["evaluate_destination"]:
        evaluation_functions.append(evaluate_destination)
        counters.append([Counter() for _ in range(n_base_locations)])
        evaluating_metrics.append("destination")
    if kwargs["evaluate_route"]:
        evaluation_functions.append(evaluate_route)
        counters.append([Counter() for _ in range(n_base_locations)])
        evaluating_metrics.append("route")
    if kwargs["evaluate_distance"]:
        evaluation_functions.append(evaluate_distance)
        counters.append(Counter())
        evaluating_metrics.append("distance")

    if len(counters) != 0:
        evaluation_functions.append(first_count)
        counters.append(Counter())
        evaluating_metrics.append("first_location")

    return evaluating_metrics, evaluation_functions, counters


def post_process_generated(generated, **kwargs):

    if len(generated) == 2:
        generated_trajs, generated_time_trajs = generated
        # generated_time_trajs = dataset.convert_time_label_trajs_to_time_trajs(generated_time_trajs)
    else:
        generated_trajs = generated
        generated_time_trajs = dataset.time_label_trajs

    # if kwargs["need_downsampling"]:
    #     original_length = len(generated_trajs)
    #     generated_trajs, indice = downsample_trajs(generated_trajs, kwargs["downsampling_dict"])
    #     # print(f"downsampled {original_length} trajectories to {len(generated_trajs)} trajectories")
    #     generated_time_trajs = [generated_time_trajs[i] for i in indice]

    # handling time is not implemented yet

    if kwargs["is_route_generator"]:
        generated_route_trajs = generated_trajs
        generated_stay_trajs = get_stay_point(generated_trajs, generated_time_trajs, kwargs["time_threshold"])
    else:
        if not kwargs["compensation"]:
            generated_route_trajs = generated_trajs
            generated_stay_trajs = generated_trajs
        else:
            generated_route_trajs, valid_ids = compensate_trajs(generated_trajs, route_db_path)
            # generated_stay_trajs = np.array(generated_trajs)[valid_ids].tolist()
            generated_stay_trajs = [traj for i, traj in enumerate(generated_trajs) if i in valid_ids]
            n_invalid += len(generated_trajs) - len(generated_route_trajs)
        
    return generated_stay_trajs, generated_route_trajs

def evaluate(generator, dataset, save_dir, logger, **kwargs):

    # n_bins = int(np.sqrt(dataset.n_locations)-2)
    # print("???")
    # if args.compensation:
    #     original_dataset_name = get_original_dataset_name(dataset)
    #     print(type(original_dataset_name), original_dataset_name)
    #     route_db_path = get_datadir() / original_dataset_name / "pair_to_route"/ f"{n_bins}_tr{args.truncate}" / "paths.db"
    #     # copy the database to ./ to avoid the latency
    #     subprocess.run(["cp", route_db_path, "./"])
    #     route_db_path = "./paths.db"
    #     print("compensating trajectories by", route_db_path)
    # else:
    #     print("not compensating trajectories")

    # n_test_locations = min(1, len(dataset.top_base_locations))
    # n_2nd_order_test_locations = min(1, len(dataset.top_2nd_order_base_locations))
    # n_test_locations = len(dataset.top_base_locations)
    # n_2nd_order_test_locations = len(dataset.top_2nd_order_base_locations)

    generator.eval()
    with torch.no_grad():
        results = {}

        if kwargs["evaluate_first_next_location"]:
            # print("DEPRECATED: evaluate_first_next_location")
            # jss = evaluate_next_location_on_test_dataset(dataset.first_next_location_counts, dataset.first_order_test_data_loader, dataset.first_counters, generator, 1)
            jss = evaluate_next_location_on_test_dataset(dataset.real_counters[dataset.evaluating_metrics_names.index("emp_next")], dataset.top_base_locations, dataset.n_locations, dataset.first_order_test_data_loader, dataset.first_counters, generator, 1)
            results["first_next_location_js"] = jss
            logger.info(f"computed divergence for first_next_location: {np.mean(jss)}")

        if kwargs["evaluate_second_next_location"] and (dataset.min_len > 2):
            jss = evaluate_next_location_on_test_dataset(dataset.second_next_location_counts, dataset.first_order_test_data_loader, dataset.first_counters, generator, 2)
            results["second_next_location_js"] = jss

        if kwargs["evaluate_second_order_next_location"] and (dataset.min_len > 2):
            print("DEPRECATED: evaluate_second_order_next_location")
            jss = evaluate_next_location_on_test_dataset(dataset.second_order_next_location_counts, dataset.second_order_test_data_loader, dataset.second_counters, generator, 2)
            results["second_order_next_location_js"] = jss

            
        # if any([kwargs["evaluate_global"], kwargs["evaluate_passing"], kwargs["evaluate_source"], kwargs["evaluate_target"], kwargs["evaluate_route"], kwargs["evaluate_destination"], kwargs["evaluate_distance"]]):

            # counters = {"global":[Counter() for _ in dataset.time_ranges], "passing": Counter(), "source": Counter(), "target": [Counter() for _ in range(n_test_locations)], "route": [Counter() for _ in range(n_test_locations)], "destination": [Counter() for _ in range(n_test_locations)], "distance": Counter(), "first_location": Counter()}
            # counters = {"passing": Counter(), "source": Counter(), "emp_next": [Counter() for _ in range(n_test_locations)], "second_emp_next": [Counter() for _ in range(n_2nd_order_test_locations)], "target": [Counter() for _ in range(n_test_locations)], "route": [Counter() for _ in range(n_test_locations)], "destination": [Counter() for _ in range(n_test_locations)], "distance": Counter(), "first_location": Counter()}
            # condition = True
        n_gene_traj = 0
        n_invalid = 0
        evaluating_metrics_names, _, counters = make_counting_functions(len(dataset.top_base_locations), **kwargs)
        while (n_gene_traj < len(dataset.references)) and dataset.counting_functions:
            mini_batch_size =  min([1000, len(dataset.references)])
            # sample mini_batch_size references from dataset.references
            sample_index = random.sample(range(len(dataset.references)), mini_batch_size)
            references = [dataset.references[i] for i in sample_index]
            time_referenfes = [dataset.time_references[i] for i in sample_index]
            # references = random.sample(dataset.references ,mini_batch_size)
            # time_referenfes = [dataset.time_references[ref] for ref in references]

            generated = generator.make_sample(references, time_referenfes, mini_batch_size)

            # post processing
            generated_stay_trajs, generated_route_trajs = post_process_generated(generated, **kwargs)

            # counting to make each distribution
            for counting_function, counter in zip(dataset.counting_functions, counters):
                counting_function(generated_stay_trajs, generated_route_trajs, dataset, counter)
                # if result is list:
                #     for result_, counter_ in zip(result, counter):
                #         counter_ += result_
                # else:
                #     counter += evaluation_function(generated_stay_trajs, dataset)

            # count first location
            # first_locations = [traj[0] for traj in generated_stay_trajs if len(traj) > 1]
            # counters[-1] += Counter(first_locations)

            # if kwargs["evaluate_global"]:
            #     for time_label in range(1, dataset.n_time_split+1):
            #         counters["global"][time_label-1] += compute_global_counts_from_time_label(generated_stay_trajs, generated_time_trajs, time_label)
            
            # if kwargs["evaluate_passing"]:
            #     counters["passing"] += count_passing_locations(generated_route_trajs)

            # if kwargs["evaluate_source"]:
            #     counters["source"] += count_source_locations(generated_stay_trajs)

            # for i, location in enumerate(dataset.top_base_locations):
            #     if kwargs["evaluate_emp_next"]:
            #         counters["emp_next"][i] += count_first_next_locations(generated_stay_trajs, location)

            #     if kwargs["evaluate_target"]:
            #         counters["target"][i] += count_target_locations(generated_stay_trajs, location)

            #     if kwargs["evaluate_destination"]:
            #         counters["destination"][i] += count_route_locations(generated_stay_trajs, location)

            #     if kwargs["evaluate_route"]:
            #         counters["route"][i] += count_route_locations(generated_route_trajs, location)

            # for i, locations in enumerate(dataset.top_2nd_order_base_locations):
            #     if kwargs["evaluate_second_emp_next"]:
            #         counters["second_emp_next"][i] += count_second_order_first_next_locations(generated_stay_trajs, locations)

            # if kwargs["evaluate_distance"]:
            #     counters["distance"] += count_distance(dataset.distance_matrix, generated_stay_trajs, dataset.n_bins_for_distance)
    
            # evaluate the same number of generated data as the ten times of that of original data
            n_gene_traj += len(generated_route_trajs)

            # save

        # save(pathlib.Path(kwargs["save_path"]) / f"evaluated_{epoch}.csv", gene_trajs)
        # print(f"saved evaluated file ({len(gene_trajs)}) to", pathlib.Path(kwargs["save_path"]) / f"evaluated_{epoch}.csv")]
        logger.info(f"generating {n_gene_traj} trajectories, there existed {n_invalid} invalid trajectories")
        # compute js
        # real_counters = dataset.real_counters
        # n_trajs = dataset.n_trajs
        # img_dir = save_dir / f"imgs_trun{kwargs['truncation']}_{kwargs['to_bin']}" / kwargs["name"]
        img_dir = save_dir / "imgs"
        img_dir.mkdir(exist_ok=True)

        first_location_counts = counters[evaluating_metrics_names.index("first_location")]
        real_first_location_counts = dataset.real_counters[dataset.evaluating_metrics_names.index("first_location")]
    
        for key, key2, counter, real_counter in zip(evaluating_metrics_names, dataset.evaluating_metrics_names, counters, dataset.real_counters):
            if key == "distance":
                n_vocabs = dataset.n_bins_for_distance
            else:
                n_vocabs = dataset.n_locations

            # evaluation of conditional metrics
            if key in ["target", "destination", "route", "emp_next"]:
                results[f"{key}_kls_eachdim"] = [compute_divergence(real_counter_, real_first_location_counts[location], counter_, first_location_counts[location], n_vocabs, save_path=img_dir / f"{key}_{i}.png", location=location) for i, (counter_, real_counter_, location) in enumerate(zip(counter, real_counter, dataset.top_base_locations))]
                results[f"{key}_jss_eachdim"] = [compute_divergence(real_counter_, real_first_location_counts[location], counter_, first_location_counts[location], n_vocabs, type="kl") for counter_, real_counter_, location in zip(counter, real_counter, dataset.top_base_locations)]
                results[f"{key}_kls_positivedim"] = [compute_divergence(real_counter_, real_first_location_counts[location], counter_, first_location_counts[location], n_vocabs, positive=True) for counter_, real_counter_, location in zip(counter, real_counter, dataset.top_base_locations)]
                results[f"{key}_jss_positivedim"] = [compute_divergence(real_counter_, real_first_location_counts[location], counter_, first_location_counts[location], n_vocabs, positive=True, type="kl") for counter_, real_counter_, location in zip(counter, real_counter, dataset.top_base_locations)]
                results[f"{key}_jss"] = [compute_divergence(real_counter_, sum(real_counter_.values()), counter_, sum(counter_.values()), n_vocabs, axis=1) for counter_, real_counter_ in zip(counter, real_counter)]
                results[f"{key}_emd"] = [compute_divergence(real_counter_, sum(real_counter_.values()), counter_, sum(counter_.values()), n_vocabs, type="emd", distance_matrix=dataset.distance_matrix) for counter_, real_counter_ in zip(counter, real_counter)]

                logger.info(f"computed divergence for {key}: {np.mean(results[f'{key}_jss'])}")
            # if key in ["target", "destination", "route", "emp_next"]:
            #     results[f"{key}_kls_eachdim"] = [compute_divergence(real_counter, n_traj, counter_, counters["first_location"][location], n_vocabs, save_path=img_dir / f"{key}_{i}.png", location=location) for i, (counter_, real_counter, n_traj, location) in enumerate(zip(counter, real_counters[key], n_trajs[key], dataset.top_base_locations))]
            #     results[f"{key}_jss_eachdim"] = [compute_divergence(real_counter, n_traj, counter_, counters["first_location"][location], n_vocabs, type="kl") for counter_, real_counter, n_traj, location in zip(counter, real_counters[key], n_trajs[key], dataset.top_base_locations)]
            #     results[f"{key}_kls_positivedim"] = [compute_divergence(real_counter, n_traj, counter_, counters["first_location"][location], n_vocabs, positive=True) for counter_, real_counter, n_traj, location in zip(counter, real_counters[key], n_trajs[key], dataset.top_base_locations)]
            #     results[f"{key}_jss_positivedim"] = [compute_divergence(real_counter, n_traj, counter_, counters["first_location"][location], n_vocabs, positive=True, type="kl") for counter_, real_counter, n_traj, location in zip(counter, real_counters[key], n_trajs[key], dataset.top_base_locations)]
            #     results[f"{key}_jss"] = [compute_divergence(real_counter, sum(real_counter.values()), counter_, sum(counter_.values()), n_vocabs, axis=1) for counter_, real_counter in zip(counter, real_counters[key])]
            #     results[f"{key}_emd"] = [compute_divergence(real_counter, sum(real_counter.values()), counter_, sum(counter_.values()), n_vocabs, type="emd", distance_matrix=dataset.distance_matrix) for counter_, real_counter, n_traj in zip(counter, real_counters[key], n_trajs[key])]
            # elif key == "second_emp_next":
            #     results[f"{key}_jss"] = [compute_divergence(real_counter, sum(real_counter.values()), counter_, sum(counter_.values()), n_vocabs, axis=1) for counter_, real_counter in zip(counter, real_counters[key])]
            # elif key == "global":
            #     results[f"{key}_kls_eachdim"] = [compute_divergence(real_counter, n_traj, counter_, n_gene_traj, n_vocabs) for counter_, real_counter, n_traj in zip(counter, real_counters[key], n_trajs[key])]
            #     results[f"{key}_kls_positivedim"] = [compute_divergence(real_counter, n_traj, counter_, n_gene_traj, n_vocabs, positive=True) for counter_, real_counter, n_traj in zip(counter, real_counters[key], n_trajs[key])]
            #     results[f"{key}_jss_eachdim"] = [compute_divergence(real_counter, n_traj, counter_, n_gene_traj, n_vocabs, type="kl") for counter_, real_counter, n_traj in zip(counter, real_counters[key], n_trajs[key])]
            #     results[f"{key}_jss_positivedim"] = [compute_divergence(real_counter, n_traj, counter_, n_gene_traj, n_vocabs, positive=True, type="kl") for counter_, real_counter, n_traj in zip(counter, real_counters[key], n_trajs[key])]
            #     results[f"{key}_jss"] = [compute_divergence(real_counter, sum(real_counter.values()), counter_, sum(counter_.values()), n_vocabs, axis=1) for counter_, real_counter in zip(counter, real_counters[key])]
            # evaluation of global metrics
            else:
                results[f"{key}_kl_eachdim"] = compute_divergence(real_counter, len(dataset.data), counter, n_gene_traj, n_vocabs)
                results[f"{key}_kl_positivedim"] = compute_divergence(real_counter, len(dataset.data), counter, n_gene_traj, n_vocabs, positive=True)
                results[f"{key}_js_eachdim"] = compute_divergence(real_counter, len(dataset.data), counter, n_gene_traj, n_vocabs, type="kl")
                results[f"{key}_js_positivedim"] = compute_divergence(real_counter, len(dataset.data), counter, n_gene_traj, n_vocabs, positive=True, type="kl")
                results[f"{key}_js"] = compute_divergence(real_counter, sum(real_counter.values()), counter, sum(counter.values()), n_vocabs, axis=1)

                logger.info(f"computed divergence for {key}: {results[f'{key}_js']}")
            # # compute js divergence
            # if key in ["target", "destination", "route"]:
            #     results[f"{key}_jss"] = []
            #     for i, (counter_, real_counter) in enumerate(zip(counter, real_counters[key])):
            #         results[f"{key}_jss"].append(compute_divergence(real_counter, sum(real_counter.values()), counter_, sum(counter_.values()), n_vocabs, axis=1))
            #         # plot_density(counter_, dataset.n_locations, img_dir / f"{key}_{i}.png", dataset.top_base_locations[i], coef=1/counters["first_location"][dataset.top_base_locations[i]])
            # elif key == "global":
            #     for i, (counter_, real_counter) in enumerate(zip(counter, real_counters[key])):
            #         results[f"{key}_jss_{i}"] = compute_divergence(real_counter, sum(real_counter.values()), counter_, sum(counter_.values()), n_vocabs, axis=1)
            #         # plot_density(counter_, dataset.n_locations, img_dir / f"{key}_{i}.png")
            # else:
            #     results[f"{key}_js"] = compute_divergence(real_counters[key], sum(real_counters[key].values()), counter, sum(counter.values()), n_vocabs, axis=1)
            #     # plot_density(counter, n_vocabs, img_dir / f"{key}.png")

    return results



def get_stay_point(generated_route_trajs, generated_time_trajs, time_threshold):
    stay_trajs = []
    for route_traj, time_traj in zip(generated_route_trajs, generated_time_trajs):
        stay_traj = []
        stay_traj.append(route_traj[0])
        for i in range(1,len(route_traj)-1):
            if (time_traj[i] >= time_threshold) and (stay_traj[-1] != route_traj[i]):
                stay_traj.append(route_traj[i])
        stay_traj.append(route_traj[-1])
        stay_trajs.append(stay_traj)
            
    return stay_trajs

def compensate_trajs(trajs, db_path):
    valid_ids = []
    new_trajs = []
    counter = 0
    for id, traj in enumerate(trajs):
        invalid_path = False
        if len(traj) == 1:
            valid_ids.append(id)
            new_trajs.append(traj)
        else:
            new_traj = [traj[0]]
            for i in range(len(traj)-1):
                edges = compensate_edge_by_map(traj[i], traj[i+1], db_path)
                invalid_path = invalid_path or (len(edges) == 0)
                new_traj.extend(edges[1:])
            if not invalid_path:
                valid_ids.append(id)
                new_trajs.append(new_traj)
            else:
                counter += 1
    # print("WARNING: n invalid trajs", counter,  "/", len(trajs))
    return new_trajs, valid_ids

def compensate_edge_by_map(from_state, to_state, db_path):

    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute(f"SELECT route FROM state_edge_to_route WHERE start_state={from_state} AND end_state={to_state}")
        edges = c.fetchone()
        if edges is None:
            # print("WARNING: path not exist", from_state, to_state)
            return []
        else:
            state_route = eval(edges[0])
            if len(state_route) == 0:
                # print("WARNING: path not exist", from_state, to_state)
                return []
            # choose the shortest one
            # shortest_route = min(state_routes, key=lambda x: len(x))
            return state_route



def compute_divergence(real_count, n_real_traj, inferred_count, n_gene_traj, n_vocabs, axis=0, positive=False, type="kl", save_path=None, location=None, distance_matrix=None):
    if n_real_traj == 0:
        print("WARNING: n_real_traj is zero")
        raise ValueError("no trajectory is evaluated")
    if n_gene_traj == 0:
        print("WARNING: n_gene_traj is zero, convert to uniform")
        inferred_count = {key: 1 for key in range(n_vocabs)}
        n_gene_traj = n_vocabs

    real_distribution = compute_distribution_from_count(real_count, n_vocabs, n_real_traj)
    if save_path is not None:
        plot_density(real_distribution, n_vocabs, save_path.parent / ("real_" + save_path.stem), anotation=location)

    inferred_distribution = compute_distribution_from_count(inferred_count, n_vocabs, n_gene_traj)
    if save_path is not None:
        plot_density(inferred_distribution, n_vocabs, save_path.parent / ("inferred_" + save_path.stem), anotation=location)

    if type == "emd":
        assert n_real_traj == sum(real_count.values()), "n_real_traj must be equal to sum(real_count.values())"
        assert n_gene_traj == sum(inferred_count.values()), "n_gene_traj must be equal to sum(inferred_count.values())"
        # compute the earth mover's distance using pyemd
        # real_count and inferred_count will be density
        true_hist = real_distribution
        inferred_hist = inferred_distribution
        # print(true_hist.shape, inferred_hist.shape, distance_matrix.shape)
        emd = pyemd.emd(inferred_hist, true_hist, distance_matrix)
        return emd

    if axis == 0:

        # compute the kl divergence on the dimensions that are positive
        real_distribution = np.stack([real_distribution, 1-real_distribution], axis=0)
        inferred_distribution = np.stack([inferred_distribution, 1-inferred_distribution], axis=0)
        # plus epsilon value to avoid inf for zero dimension
        inferred_distribution[inferred_distribution == 0] = 1e-10

        if positive:
            # filter out the negative dimensions
            positive_indices = np.where(real_distribution[0] > 0)[0]
            real_distribution = real_distribution[:, positive_indices]
            inferred_distribution = inferred_distribution[:, positive_indices]

        # this is for debug
        if scipy.stats.entropy(real_distribution, inferred_distribution, axis=0).sum() == float("inf"):
            for i in range(n_vocabs):
                if scipy.stats.entropy(real_distribution[:, i], inferred_distribution[:, i], axis=0) == float("inf"):
                    print(i)
                    print(real_distribution[:, i])
                    print(inferred_distribution[:, i])
                    if i in real_count:
                        print(real_count[i])
                    else:
                        print("not in real")
                    if i in inferred_count:
                        print(inferred_count[i])
                        print("not in inferred")
                    print(n_real_traj, real_count)
                    print(n_gene_traj, inferred_count)
                    raise ValueError("inf")

        if type == "kl":
            return scipy.stats.entropy(real_distribution, inferred_distribution, axis=0).sum()
        else:
            return (jensenshannon(real_distribution, inferred_distribution, axis=0)**2).sum()
    else:
        # real_count and inferred_count will be the probability distributions
        assert n_real_traj == sum(real_count.values()), "n_real_traj must be equal to sum(real_count.values())"
        assert n_gene_traj == sum(inferred_count.values()), "n_gene_traj must be equal to sum(inferred_count.values())"
        return jensenshannon(real_distribution, inferred_distribution)**2



def make_target_distributions_of_all_layers(target_distribution, tree):
    # from the location distribution on the all states (i.e., leafs), make the target distribution of all layers
    # target_distribution: (batch_size, n_locations)
    tree._register_count_to_complete_graph(target_distribution)
    distributions = [target_distribution]
    for depth in list(range(tree.max_depth))[1:][::-1]:
        nodes = tree.get_nodes(depth)
        for node in nodes:
            node.count = 0
            for child in node.children:
                node.count += child.count
        distribution = {node: node.count for node in nodes}
        # sort according to node.oned_coordinate
        distribution = torch.stack([v for _, v in sorted(distribution.items(), key=lambda item: item[0].oned_coordinate)], dim=1)
        distributions.append(distribution)
    return distributions[::-1]


def compute_distribution_js_for_each_depth(distribution, target_distribution):
    next_location_js_for_all_depth = []
    n_locations = distribution.shape[-1]
    tree = construct_default_quadtree(int(np.sqrt(n_locations))-2)
    tree.make_self_complete()
    target_next_location_distribution_for_all_depth = make_target_distributions_of_all_layers(torch.tensor(target_distribution).view(-1,n_locations), tree)
    generated_next_location_distribution_for_all_depth = make_target_distributions_of_all_layers(torch.tensor(distribution).view(-1,n_locations), tree)
    for depth in range(1, tree.max_depth+1):
        next_location_js_for_all_depth.append(jensenshannon(target_next_location_distribution_for_all_depth[depth-1], generated_next_location_distribution_for_all_depth[depth-1], axis=1)**2)
    return np.stack(next_location_js_for_all_depth, axis=1).tolist()


def evaluate_next_location_on_test_dataset(next_location_counts, top_k_locations, n_locations, data_loader, counters, generator, target_index):
    # next_location_distributions = {key: noise_normalize(next_location_count) for key, next_location_count in next_location_counts.items()}
    # print(next_location_counts)
    next_location_distributions = {key: compute_distribution_from_count(next_location_count, n_locations, sum(next_location_count.values())) for key, next_location_count in zip(top_k_locations, next_location_counts)}
    jss = []

    outputs = []
    for mini_batch in data_loader:
        if hasattr(generator, "transition_matrix"):
            input_locations = mini_batch["input"]
            output = torch.exp(generator(input_locations[:, target_index]))
        else:
            device = next(iter(generator.parameters())).device
            input_locations = mini_batch["input"].to(device)
            references = [tuple(v) for v in mini_batch["reference"]]
            input_times = mini_batch["time"].to(device)
            # output = generator([input_locations, input_times], references)[0]
            output, _ = generator([input_locations, input_times])[0]
            output = output[-1] if type(output) == list else output
            output = torch.exp(output).cpu().detach().numpy()[:, target_index].tolist()
            outputs.extend(output)
    
    cursor = 0
    for target, n_test_data in counters.items():
        output = outputs[cursor:cursor+n_test_data]
        inferred_distribution = np.mean(output, axis=0)
        target_distribution = next_location_distributions[target]
        jss.append(compute_distribution_js_for_each_depth(inferred_distribution, target_distribution))

        cursor += n_test_data

    return jss


def compute_destination_count(trajs, source_location):
    # find the trajs that start from the source location
    trajs_from_source = [traj for traj in trajs if traj[0] == source_location]
    # compute the route distribution
    route_locations = []
    for traj in trajs_from_source:
        route_locations.append(traj[-1])
    route_count = Counter(route_locations)
    return route_count

def count_source_locations(trajs):
    start_locations = []
    for traj in trajs:
        start_locations.append(traj[0])
    return Counter(start_locations)

def count_passing_locations(trajs):
    # count the appearance of locations
    passing_locations = []
    for traj in trajs:
        passing_locations.extend(list(set(traj[1:])))
    return Counter(passing_locations)

def count_target_locations(trajs, source_location):
    trajs = [traj for traj in trajs if traj[0] == source_location and len(traj) > 1]
    # trajs = [traj for traj in trajs if traj[0] == source_location]
    target_locations = []
    for traj in trajs:
        target_locations.append(traj[-1])
    return Counter(target_locations)

def compute_distribution_from_count(count, n_locations, n_trajs):
    distribution = np.zeros(n_locations)
    for key, value in count.items():
        distribution[key] = value
    return distribution / n_trajs

# compute the route distribution
# i.e., given the source location, compute the probability of each location passing through
def count_route_locations(trajs, source_location):
    # find the trajs that start from the source location
    trajs_from_source = [traj for traj in trajs if traj[0] == source_location]
    # compute the route distribution
    route_locations = []
    for traj in trajs_from_source:
        route_locations_ = list(set(traj[1:]) - set([source_location]))
        route_locations.extend(route_locations_)
    route_count = Counter(route_locations)
    return route_count

def compute_distance(distance_matrix, traj):
    distance = 0
    for i in range(len(traj)-1):
        distance += distance_matrix[traj[i], traj[i+1]]
    return distance

def compute_distances(distance_matrix, trajs):
    distances = []
    for traj in trajs:
        distance = 0
        for i in range(len(traj)-1):
            distance += distance_matrix[traj[i]][traj[i+1]]
        distances.append(distance)
    return distances

def count_distance(distance_matrix, trajs, n_bins):
    distances = compute_distances(distance_matrix, trajs)
    # make histogram using n_bins
    hist, _ = np.histogram(distances, bins=n_bins)
    # compute prob
    count = {key:v for key, v in enumerate(hist)}
    return count


def count_first_next_locations(trajs, source_location):
    trajs = [traj for traj in trajs if traj[0] == source_location and len(traj) > 1]
    # count the appearance of locations
    first_next_locations = []
    for traj in trajs:
        # if traj[0] == source_location:
        first_next_locations.append(traj[1])
    return Counter(first_next_locations)

def count_second_order_first_next_locations(trajs, source_location):
    # count the appearance of locations
    second_order_first_next_locations = []
    for traj in trajs:
        if len(traj) < 3:
            continue
        if traj[0] == source_location[0] and traj[1] == source_location[1]:
            second_order_first_next_locations.append(traj[2])
    return Counter(second_order_first_next_locations)


# def compute_next_location_distribution(target, trajectories, n_locations):
#     # compute the next location probability for each location
#     counts = compute_next_location_count(target, trajectories, n_locations)
#     summation = sum(counts)
#     if summation == 0:
#         return None
#     distribution = [count/summation for count in counts]
#     return distribution

def compute_next_location_count(target, trajectories, n_locations, target_index=0):
    # compute the count of next location for each location
    if target_index == 0:
        count = Counter([trajectory[i+1] for trajectory in trajectories for i in range(len(trajectory)-1) if trajectory[i]==int(target)])
    elif target_index == 1:
        trajectories = [trajectory for trajectory in trajectories if len(trajectory)>1]
        count = Counter([trajectory[1] for trajectory in trajectories if trajectory[0]==target])
    elif target_index == 2:
        trajectories = [trajectory for trajectory in trajectories if len(trajectory)>2]
        count = Counter([trajectory[2] for trajectory in trajectories if trajectory[1]==target])
    counts = []
    for i in range(n_locations):
        if i not in count:
            counts.append(0)
        else:
            counts.append(count[i])
    return counts



def make_next_location_count(dataset, target_index, order=1):
    if order == 1:
        # print(f"compute {target_index} next location count")
        # compute the next location probability for each location
        next_location_counts = {}
        for label, traj in zip(dataset.labels, dataset.data):
            reference = dataset.label_to_reference[label]

            if target_index == 0:
                # count next location by the marginal way
                for i in range(1,len(reference)):
                    if not (reference[i] == max(reference[:i+1])):
                        continue
                    if traj[i-1] not in next_location_counts:
                        next_location_counts[traj[i-1]] = [0 for _ in range(dataset.n_locations)]
                    next_location_counts[traj[i-1]][traj[i]] += 1
            elif target_index == 1:
                # count next location by the first next location
                if len(reference) < 2:
                    continue
                if traj[0] not in next_location_counts:
                    next_location_counts[traj[0]] = [0 for _ in range(dataset.n_locations)]
                next_location_counts[traj[0]][traj[1]] += 1
            elif target_index == 2:
                # count next location by the second next location
                if len(reference) < 3:
                    continue
                if reference[2] != 2:
                    continue
                if traj[0] not in next_location_counts:
                    next_location_counts[traj[0]] = [0 for _ in range(dataset.n_locations)]
                next_location_counts[traj[0]][traj[2]] += 1

    elif order == 2:
        # print(f"compute {target_index} first second order next location count")
        # compute the next location probability for each location
        next_location_counts = {}
        for label, traj in zip(dataset.labels, dataset.data):
            reference = dataset.label_to_reference[label]
            if len(reference) < 3:
                continue
            if reference[2] != 2:
                continue

            if (traj[0], traj[1]) not in next_location_counts:
                next_location_counts[(traj[0], traj[1])] = [0 for _ in range(dataset.n_locations)]
            next_location_counts[(traj[0], traj[1])][traj[2]] += 1

    return next_location_counts




# compute each count for each time split
# ex) traj [3,24,25,3], time_traj [1,3,3,4]
# at time 1~3 -> 3
# at time 3~4 -> 24,25
# at time 4~ -> 3
# note at time 2 -> 3
def compute_global_counts_from_time_label(trajs, time_label_trajs, time_label):
    # find the locations at time
    def locations_at_time(traj, time_label_traj, time_label):
        # if time_label in time_label_traj, return the locations at time
        # else, return the final location of the index that is the closest to the time_label
        assert time_label >= 1, "time_label should be larger than 1 because 0 is the start signal"
        if time_label in time_label_traj:
            indice = [i for i, t in enumerate(time_label_traj) if t == time_label]
            return [traj[i] for i in indice]
        else:
            indice = [i for i, t in enumerate(time_label_traj) if t <= time_label]
            if len(indice) == 0:
                return []
            else:
                return [[traj[i] for i in indice][-1]]

    locations = []
    for traj, time_label_traj in zip(trajs, time_label_trajs):
        locations.extend(locations_at_time(traj, time_label_traj, time_label))

    location_count = Counter(locations)

    return location_count



def compute_auxiliary_information(dataset, save_dir, test_thresh, logger, **kwargs):
    save_dir = pathlib.Path(save_dir)
    img_dir = save_dir.parent / f"imgs"
    img_dir.mkdir(exist_ok=True)

    # compute top_base_locations
    dataset.first_locations = [trajectory[0] for trajectory in dataset.data if len(trajectory) > 1]
    dataset.first_location_counts = Counter(dataset.first_locations)
    dataset.route_first_locations = [trajectory[0] for trajectory in dataset.route_data if len(trajectory) > 1]
    dataset.route_first_location_counts = Counter(dataset.route_first_locations)
    dataset.second_order_first_locations = [tuple(trajectory[:2]) for trajectory in dataset.data if len(trajectory) > 2]
    dataset.second_order_first_locations_counts = Counter(dataset.second_order_first_locations)
    # find locations whose count is larger than test_thresh and sort them
    test_thresh = test_thresh
    test_locations = {location:count for location, count in dataset.first_location_counts.items() if count > test_thresh}
    max_test_loc = 20
    if len(test_locations) > max_test_loc:
        logger.info(f"WARNING number of test locations is larger than {max_test_loc}")
        # find the location that has the largest count until the number of test locations become 20
        test_locations = {k: v for k, v in sorted(dataset.first_location_counts.items(), key=lambda item: item[1], reverse=True)[:max_test_loc]}
    if len(test_locations) == 0:
        logger.info("WARNING no test location is found")
        # find the location that has the largest count until the number of test locations become 20
        test_locations = {k: v for k, v in sorted(dataset.first_location_counts.items(), key=lambda item: item[1], reverse=True)[:1]}
    logger.info(f"number of test locations: {len(test_locations)}; {test_locations}")

    test_thresh_second = 50
    second_order_test_locations = {locations:count for locations, count in dataset.second_order_first_locations_counts.items() if count > test_thresh_second}
    max_test_loc = 20
    if len(second_order_test_locations) > max_test_loc:
        logger.info(f"WARNING number of 2nd order test locations is larger than {max_test_loc}")
        # find the location that has the largest count until the number of test locations become 20
        second_order_test_locations = {k: v for k, v in sorted(dataset.second_order_first_locations_counts.items(), key=lambda item: item[1], reverse=True)[:max_test_loc]}
    if len(second_order_test_locations) == 0:
        logger.info("WARNING no 2nd order test location is found")
        # find the location that has the largest count
        second_order_test_locations = {k: v for k, v in sorted(dataset.second_order_first_locations_counts.items(), key=lambda item: item[1], reverse=True)[:1]}
    logger.info(f"number of 2nd order test locations: {len(second_order_test_locations)}; {second_order_test_locations}")

    dataset.top_base_locations = sorted(test_locations, key=lambda x: dataset.first_location_counts[x], reverse=True)
    dataset.top_route_base_locations = sorted(test_locations, key=lambda x: dataset.route_first_location_counts[x], reverse=True)
    dataset.top_2nd_order_base_locations = sorted(second_order_test_locations, key=lambda x: dataset.first_location_counts[x], reverse=True)
    # compute the next location counts
    dataset.next_location_counts = make_next_location_count(dataset, 0)
    dataset.first_next_location_counts = make_next_location_count(dataset, 1)
    dataset.second_next_location_counts = make_next_location_count(dataset, 2)
    dataset.second_order_next_location_counts = make_next_location_count(dataset, 0, order=2)

    make_first_order_test_data_loader(dataset, 20)
    
    # compute the global counts
    real_global_counts = []
    for time in range(1,dataset.n_time_split+1):
        real_global_count = Counter({location:0 for location in range(dataset.n_locations)})
        real_global_count.update(compute_global_counts_from_time_label(dataset.data, dataset.time_label_trajs, time))
        real_global_count = list(real_global_count.values())
        real_global_counts.append(real_global_count)
        if sum(real_global_count) == 0:
            logger.info(f"no location at time {time}")
            continue
        plot_density(real_global_count, dataset.n_locations, save_dir.parent / f"imgs" / f"real_global_distribution_{int(time)}.png")

    # global_counts_path = save_dir.parent / f"global_count.json"
    # # save the global counts
    # with open(global_counts_path, "w") as f:
    #     json.dump(real_global_counts, f)
    # dataset.global_counts = real_global_counts

    # # make a list of labels
    # label_list = [dataset.format_to_label[traj_to_format(trajectory)] for trajectory in dataset.data]
    # label_count = Counter({label:0 for label in dataset.label_to_format.keys()})
    # label_count.update(label_list)
    # reference_distribution = {dataset.label_to_reference[label]: count for label, count in label_count.items()}
    
    # # compute time distribution
    # time_label_count = Counter(dataset.time_label_trajs)
    # time_distribution = {label: time_label_count[label] / len(dataset.time_label_trajs) for label in time_label_count.keys()}
    dataset.distance_matrix = np.load(get_datadir() / str(dataset)  / f"distance_matrix_bin{int(np.sqrt(dataset.n_locations)) -2}.npy")

    dataset.evaluating_metrics_names, dataset.counting_functions, dataset.real_counters = make_counting_functions(len(dataset.top_base_locations), **kwargs)
    logger.info(f"evaluating metrics: {dataset.evaluating_metrics_names}")
    # counting to make each distribution
    for counting_function, counter in zip(dataset.counting_functions, dataset.real_counters):
        counting_function(dataset.data, dataset.route_data, dataset, counter)

    # dataset.evaluating_metrics = []
    # dataset.real_counters = []
    # dataset.n_trajs = []
    # if kwargs["evaluate_passing"]:
    #     dataset.evaluating_metrics.append("passing")
    #     dataset.real_counters.append(count_passing_locations(dataset.data))
    #     dataset.n_trajs.append(len(dataset.data))
    # if kwargs["evaluate_source"]:
    #     dataset.evaluating_metrics.append("source")
    #     dataset.real_counters.append(count_source_locations(dataset.data))
    #     dataset.n_trajs.append(len(dataset.data))
    # if kwargs["evaluate_emp_next"]:
    #     dataset.evaluating_metrics.append("emp_next")
    #     dataset.real_counters.append([count_first_next_locations(dataset.data, location) for location in dataset.top_base_locations])
    #     dataset.n_trajs.append(sum(dataset.first_location_counts.values()))
    # # if kwargs["evaluate_second_emp_next"]:
    # #     evaluation_functions.append(count_second_order_first_next_locations)
    # if kwargs["evaluate_target"]:
    #     dataset.evaluating_metrics.append("target")
    #     dataset.real_counters.append([count_target_locations(dataset.data, location) for location in dataset.top_base_locations])
    #     dataset.n_trajs.append([dataset.first_location_counts[location] for location in dataset.top_base_locations])
    # if kwargs["evaluate_destination"]:
    #     dataset.evaluating_metrics.append("destination")
    #     dataset.real_counters.append([compute_destination_count(dataset.data, location) for location in dataset.top_base_locations])
    #     dataset.n_trajs.append([dataset.first_location_counts[location] for location in dataset.top_base_locations])
    # if kwargs["evaluate_route"]:
    #     dataset.evaluating_metrics.append("route")
    #     dataset.real_counters.append([count_route_locations(dataset.route_data, location) for location in dataset.top_base_locations])
    #     dataset.n_trajs.append([dataset.route_first_location_counts[location] for location in dataset.top_base_locations])
    # if kwargs["evaluate_distance"]:
    #     dataset.evaluating_metrics.append("distance")
    #     dataset.real_counters.append(count_distance(dataset.distance_matrix, dataset.data, dataset.n_bins_for_distance))
    #     dataset.n_trajs.append(len(dataset.data))

    # # compute counters
    # dataset.real_counters = {}
    # for metric, count_function in evaluating_metrics:
    # dataset.real_counters["global"] = [Counter({key:count for key, count in enumerate(global_count)}) for global_count in real_global_counts]
    # dataset.real_counters["passing"] = count_passing_locations(dataset.route_data)
    # dataset.real_counters["source"] = count_source_locations(dataset.data)
    # dataset.real_counters["target"] = [count_target_locations(dataset.data, location) for location in dataset.top_base_locations]
    # dataset.real_counters["route"] = [count_route_locations(dataset.route_data, location) for location in dataset.top_base_locations]
    # dataset.real_counters["destination"] = [count_route_locations(dataset.data, location) for location in dataset.top_base_locations]
    # dataset.real_counters["emp_next"] = [count_first_next_locations(dataset.data, location) for location in dataset.top_base_locations]
    # dataset.real_counters["second_emp_next"] = [count_second_order_first_next_locations(dataset.data, locations) for locations in dataset.top_2nd_order_base_locations]
    # logger.info("load distance matrix from {}".format(get_datadir() / str(dataset)  / f"distance_matrix_bin{int(np.sqrt(dataset.n_locations)) -2}.npy"))
    # try:
    #     dataset.distance_matrix = np.load(get_datadir() / str(dataset)  / f"distance_matrix_bin{int(np.sqrt(dataset.n_locations)) -2}.npy")
    #     dataset.real_counters["distance"] = count_distance(dataset.distance_matrix, dataset.data, dataset.n_bins_for_distance)
    # except:
    #     print("WARNING: distance matrix is not found", get_datadir() / str(dataset)  / f"distance_matrix_bin{int(np.sqrt(dataset.n_locations)) -2}.npy")

    # compute n_trajs
    # dataset.n_trajs = {}
    # dataset.n_trajs["global"] = [len(dataset.data) for global_count in real_global_counts]
    # dataset.n_trajs["passing"] = len(dataset.data)
    # dataset.n_trajs["source"] = len(dataset.data)
    # dataset.n_trajs["target"] = [dataset.first_location_counts[location] for location in dataset.top_base_locations]
    # dataset.n_trajs["route"] = [dataset.route_first_location_counts[location] for location in dataset.top_base_locations]
    # dataset.n_trajs["destination"] = [dataset.first_location_counts[location] for location in dataset.top_base_locations]
    # dataset.n_trajs["distance"] = len(dataset.data)
    # dataset.n_trajs["emp_next"] = [sum(counter.values()) for counter in dataset.real_counters["emp_next"]]
    # dataset.n_trajs["second_emp_next"] = [sum(counter.values()) for counter in dataset.real_counters["second_emp_next"]]

    # plot the counts
    # for key, counter in dataset.real_counters.items():
    #     if key == "global":
    #         for i, count in enumerate(counter):
    #             plot_density(count, dataset.n_locations, img_dir / f"real_{key}_distribution_{int(i)}.png")
    #     elif key in ["target", "destination", "route"]:
    #         for i, count in enumerate(counter[:test_thresh]):
    #             plot_density(count, dataset.n_locations, img_dir / f"real_{key}_distribution_{int(i)}.png", dataset.top_base_locations[i], coef=1/dataset.n_trajs[key][i])
    #     elif key == "distance":
    #         plot_density(counter, dataset.n_bins_for_distance, img_dir / f"real_{key}_distribution.png")
    #     else:
    #         plot_density(counter, dataset.n_locations, img_dir / f"real_{key}_distribution.png")
    
    # send(img_dir, parent=True)


    # return locations, next_location_counts, first_next_location_counts, real_global_counts, label_count, time_distribution, reference_distribution



def make_second_order_test_data_loader(dataset, n_test_locations):

    from dataset import TrajectoryDataset

    second_order_next_location_counts = dataset.second_order_next_location_counts
    n_test_locations = min(n_test_locations, len(second_order_next_location_counts))
    top_second_order_base_locations = sorted(second_order_next_location_counts, key=lambda x: sum(second_order_next_location_counts[x]), reverse=True)[:n_test_locations]

    # retrieving the trajectories that start with the first_location_counts
    counters = {}
    trajs = []
    time_trajs = []
    first_second_locations = top_second_order_base_locations[:n_test_locations]
    for first_location in first_second_locations:
        trajs_for_the_first_location = []
        time_trajs_for_the_first_location = []
        for traj, time_traj in zip(dataset.data, dataset.time_data):
            if len(traj) > 2:
                if traj[0] == first_location[0] and traj[1] == first_location[1]:
                    trajs_for_the_first_location.append(traj)
                    time_trajs_for_the_first_location.append(time_traj)
        counters[first_location] = len(trajs_for_the_first_location)
        trajs.extend(trajs_for_the_first_location)
        time_trajs.extend(time_trajs_for_the_first_location)
    
    print(f"number of test trajectories: {len(trajs)}")
    print(f"number of test trajectories that start with: {counters}")

    if len(trajs) == 0:
        print("no trajectory (>2) is found")
        dataset.second_order_test_data_loader = None
        dataset.second_counters = None
        return None, None
    else:

        second_order_test_dataset = TrajectoryDataset(trajs, time_trajs, dataset.n_locations, dataset.n_time_split)
        second_order_test_data_loader = torch.utils.data.DataLoader(second_order_test_dataset, num_workers=0, shuffle=False, pin_memory=True, batch_size=100, collate_fn=second_order_test_dataset.make_padded_collate())
        
        dataset.second_order_test_data_loader = second_order_test_data_loader
        dataset.second_counters = counters
        return second_order_test_data_loader, counters

def make_first_order_test_data_loader(dataset, n_test_locations):

    from dataset import TrajectoryDataset

    top_base_locations = dataset.top_base_locations
    n_test_locations = min(n_test_locations, len(top_base_locations))

    # retrieving the trajectories that start with the first_location_counts
    counters = {}
    trajs = []
    time_trajs = []
    first_locations = top_base_locations
    for first_location in first_locations:
        trajs_for_the_first_location = []
        time_trajs_for_the_first_location = []
        for traj, time_traj in zip(dataset.data, dataset.time_data):
            if traj[0] == first_location and len(traj) > 1:
                trajs_for_the_first_location.append(traj)
                time_trajs_for_the_first_location.append(time_traj)
        counters[first_location] = len(trajs_for_the_first_location)
        trajs.extend(trajs_for_the_first_location)
        time_trajs.extend(time_trajs_for_the_first_location)
    
    print(f"number of test trajectories: {len(trajs)}")
    print(f"number of test trajectories that start with: {counters}")

    test_dataset = TrajectoryDataset(trajs, time_trajs, dataset.n_locations, dataset.n_time_split)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0, shuffle=False, pin_memory=True, batch_size=100, collate_fn=test_dataset.make_padded_collate())

    dataset.first_order_test_data_loader = test_data_loader
    dataset.first_counters = counters
    return test_data_loader, counters

# def compute_global_counts(trajectories, real_time_traj, time, n_locations, time_to_label):
#     def location_at_time(trajectory, time_traj, t):
#         if t == 0:
#             return int(trajectory[0])

#         label = time_to_label(t)
#         time_label_traj = [time_to_label(time) for time in time_traj]
#         if label not in time_label_traj:
#             return None
#         elif time_label_traj.index(label) == len(trajectory):
#             return None
#         else:
#             return trajectory[time_label_traj.index(label)]
            
#     locations = []
#     count = 0
#     for trajectory, time_traj in zip(trajectories, real_time_traj):
#         if 1+len(trajectory) != len(time_traj):
#             # print("BUG, NEED TO BE FIXED", trajectory, time_traj)
#             count += 1
#         else:
#             location = location_at_time(trajectory, time_traj, time)
#             if location is not None:
#                 locations.append(location)

#     # count each location
#     location_count = Counter(locations)
#     return location_count

# def compute_global_distribution(trajectories, real_time_traj, time, n_locations, time_to_label):
#     global_counts = compute_global_counts(trajectories, real_time_traj, time, n_locations, time_to_label)
#     global_distribution = compute_distribution_from_count(global_counts, n_locations)
#     return global_distribution

class MTNetGeneratorMock():
    '''
    This generator returns the trajectory in the given path
    '''
    def __init__(self, traj_path, time_traj_path, dataset, n_bins, random=False):
        original_dataset_name = get_original_dataset_name(dataset)
        with open("config.json", "r") as f:
            config = json.load(f)["latlon"][original_dataset_name]
        lat_range = config["lat_range"]
        lon_range = config["lon_range"]
        edge_to_state_pair, _ = self.make_edge_to_state_pair(get_datadir() / dataset / "raw", lat_range, lon_range, n_bins)
        self.trajs, self.time_trajs = self.convert_to_state(traj_path, time_traj_path, edge_to_state_pair)
        self.cursor = 0
        self.random = random


    def make_edge_to_state_pair(self, data_path, lat_range, lon_range, n_bins):
        from grid import Grid
        import os
        import shapely.wkt


        print("use grid of ", lat_range, lon_range, n_bins)
        ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
        grid = Grid(ranges)
        # edge is a tuple of states (first state, last state)
        # 1,0,0,0,LINESTRING"(39.72916666666667 116.14250000000001,39.72916666666667 116.14250000000001)"
        # first latlon (39.72916666666667 116.14250000000001) -> first state
        # last latlon (39.72916666666667 116.14250000000001) -> last state
        edge_id_to_state_pair = {}
        print("convert to original by", os.path.join(data_path, "edge_property.txt"))
        with open(os.path.join(data_path, "edge_property.txt"), "r") as f:
            for i, line in enumerate(f):
                # load wkt by shapely
                wkt = ",".join(line.split(",")[4:])[1:-1]
                wkt = shapely.wkt.loads(wkt)
                # get the first lon,lat
                from_lonlat = wkt.coords[0]
                # get the last lon,lat
                to_lonlat = wkt.coords[-1]
                from_state = grid.latlon_to_state(*from_lonlat[::-1])
                if from_state == None:
                    print(*from_lonlat[::-1], line)
                    raise
                to_state = grid.latlon_to_state(*to_lonlat[::-1])
                if to_state == None:
                    print("w", line)
                    raise
                edge = (from_state, to_state)
                edge_id_to_state_pair[i+1] = edge
        
        return edge_id_to_state_pair, grid

    def convert_to_state(self, traj_path, time_path, edge_id_to_state_pair):

        from data_pre_processing import compless

        # load data
        trajs = []
        with open(pathlib.Path(traj_path), "r") as f:
            for line in f:
                traj = [int(vocab) for vocab in line.split(",")]
                # traj = [int(vocab) for vocab in line.strip().split(" ")]
                # remove 0s at the end
                traj = [vocab for vocab in traj if vocab != 0]
                trajs.append(traj)

        time_trajs = []
        with open(pathlib.Path(time_path), "r") as f:
            for line in f:
                time_traj = [int(vocab) for vocab in line.split(",")]
                time_trajs.append(time_traj)

        new_trajs = []
        for traj in trajs:
            new_traj = []
            for i in range(len(traj)):
                if i == 0:
                    new_traj.append(edge_id_to_state_pair[traj[i]][0])
                    new_traj.append(edge_id_to_state_pair[traj[i]][1])
                else:
                    new_traj.append(edge_id_to_state_pair[traj[i]][1])
            new_trajs.append(new_traj)

        complessed_trajs = []
        complessed_time_trajs = []
        for traj, time_traj in zip(new_trajs, time_trajs):
            complessed_traj, complessed_time_traj = compless(traj, time_traj, cost=True)
            complessed_trajs.append(complessed_traj)
            complessed_time_trajs.append(complessed_time_traj)


        return complessed_trajs, complessed_time_trajs


    def eval(self):
        pass

    def train(self):
        pass

    def make_sample(self, references, mini_batch_size):
        '''
        return the mini_batch_size trajectories in the given path
        '''
        if self.random:
            if len(self.trajs)-mini_batch_size == 0:
                self.cursor = 0
            else:
                self.cursor = np.random.randint(0, len(self.trajs)-mini_batch_size)
        else:
            self.cursor += mini_batch_size
            if self.cursor >= len(self.trajs)+mini_batch_size:
                self.cursor = mini_batch_size
        return self.trajs[self.cursor-mini_batch_size:self.cursor], self.time_trajs[self.cursor-mini_batch_size:self.cursor]


# class Namespace():
    # pass

# def set_args(run_args):

#     args = Namespace()
#     args.ablation = run_args.ablation
#     args.evaluate_global = False and (not args.ablation)
#     args.evaluate_passing = True and (not args.ablation)
#     args.evaluate_source = False and (not args.ablation)
#     args.evaluate_target = True and (not args.ablation)
#     args.evaluate_route = True and (not args.ablation)
#     args.evaluate_destination = True and (not args.ablation)
#     args.evaluate_distance = True and (not args.ablation)
#     args.evaluate_emp_next = True and (not args.ablation)
#     args.evaluate_second_emp_next = True and (not args.ablation)
#     args.evaluate_first_next_location = True and (training_setting["network_type"] in ["hiemrnet", "baseline"])
#     args.evaluate_second_next_location = True and (training_setting["network_type"] in ["hiemrnet", "baseline"])
#     args.evaluate_second_order_next_location = False and (training_setting["network_type"] in ["hiemrnet", "baseline"])
#     args.dataset = dataset_name
    
#     args.eval_initial = True
#     args.n_test_locations = 30
#     args.n_split = 5
#     # this is not used
#     args.batch_size = 100
#     args.route_generator = False
#     args.time_threshold = 10
#     args.eval_interval = run_args.eval_interval
#     args.test_thresh = run_args.test_thresh

#     # args.time_threshold = run_args.time_threshold
#     args.route_generator = (training_setting["network_type"] == "mtnet")
#     args.compensation = (dataset_name in ["chengdu", "geolife_mm"]) and (not args.route_generator)
#     args.truncate = run_args.truncate
#     # if run_args.location_threshold == 0 and run_args.time_threshold == 0:
#         # args.compensation = False
#         # args.route_generator = True
    
#     # args.to_bin = run_args.n_bins

#     return args

def run(**kwargs):
    from main import construct_dataset, construct_generator

    device = torch.device(f"cuda:{kwargs['cuda_number']}" if torch.cuda.is_available() else "cpu")

    # find models
    model_dir = make_model_dir(**kwargs)
    if model_dir.stem.startswith("model_"):
        model_paths = [model_dir]
        model_dir = model_dir.parent
    else:
        # find the models whose name stats with model_i
        model_paths = list(model_dir.glob("model_*"))
    model_paths = sorted(model_paths, key=lambda x: int(x.stem.split("_")[-1]))
    logger = set_logger(__name__, model_dir / "eval.log")
    logger.info(f"evaluate models: {model_paths}")

    # make dataset
    save_name = make_save_name(**kwargs)
    training_data_dir = make_training_data_path(save_name=save_name, **kwargs)
    dataset = construct_dataset(training_data_dir, None, kwargs["n_split"])
    compute_auxiliary_information(dataset, model_dir, kwargs["test_threshold"], logger, **kwargs)

    # evaluation
    for i, model_path in enumerate(model_paths):
        # skip according to the interval
        if i % kwargs["evaluation_interval"] != 0:
            continue

        logger.info(f"evaluate {model_path}")

        # load model
        if kwargs["model_name"] in ["hrnet", "baseline"]:
            # pretraining_network, _ = construct_pretraining_network(kwargs["clustering"], kwargs["model_name"], dataset.n_locations, kwargs["memory_dim"], kwargs["memory_hidden_dim"], kwargs["location_embedding_dim"], kwargs["multilayer"], kwargs["consistent"], logger)
            # if hasattr(pretraining_network, "remove_class_to_query"):
            #     pretraining_network.remove_class_to_query()
            generator = construct_generator(kwargs["model_name"], dataset.n_locations, dataset.n_time_split+1, kwargs["location_embedding_dim"], kwargs["time_embedding_dim"], kwargs["memory_hidden_dim"], kwargs["multitask"], kwargs["consistent"])
            generator.load_state_dict(torch.load(model_path, map_location=device))
            generator = generator.to(device)

        # evaluate the model
        results = evaluate(generator, dataset, model_dir, logger, **kwargs)

        # save the result
        result_save_path = model_dir / f"evaluated_{i}_co.json" if kwargs["consistent"] else model_dir / f"evaluated_{i}.json"
        
        logger.info("save result to " + str(result_save_path))
        with open(result_save_path, "w") as f:
            json.dump(results, f)



# if __name__ == "__main__":
#     from main import construct_dataset, construct_generator, construct_meta_network
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_dir', type=str)

#     # for ground truth
#     # parser.add_argument('--location_threshold', type=int)
#     # parser.add_argument('--time_threshold', type=int)
#     # parser.add_argument('--n_bins', type=int)
#     parser.add_argument('--eval_data_dir', type=str)
#     parser.add_argument('--seed', type=int)
#     parser.add_argument('--truncate', type=int)
#     parser.add_argument('--test_thresh', type=int)
#     parser.add_argument('--eval_interval', type=int)

#     # parser.add_argument('--server', action="store_true")
#     parser.add_argument('--ablation', action="store_true")
#     run_args = parser.parse_args()

#     logger = set_logger(__name__, "./log.log")

#     # if run_args.server:
#         # get(run_args.model_dir, parent=True)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model_dir = pathlib.Path(run_args.model_dir)
    
#     if model_dir.stem.startswith("model_"):
#         model_paths = [model_dir]
#         model_dir = model_dir.parent
#     else:
#         # find the models whose name stats with model_i
#         model_paths = model_dir.glob("model_*")
    
#     # get(model_dir / "params.json")
#     with open(model_dir / "params.json", "r") as f:
#         training_setting = json.load(f)
#     (model_dir / "imgs").mkdir(exist_ok=True)
#     # data_name = training_setting["data_name"]
#     # n_bins = int(training_setting["training_data_name"].split("_")[2].split("bin")[1])

#     # if "training_data_name" in training_setting:
#         # data_path = get_datadir() / training_setting["dataset"] / training_setting["data_name"] / training_setting["training_data_name"]
#     # else:

#     training_data_dir = pathlib.Path(training_setting["training_data_dir"])
#     with open(training_data_dir / "params.json", "r") as f:
#         data_setting = json.load(f)
#     n_bins = data_setting["n_bins"]
#     dataset_name = data_setting["dataset"]
#     # if run_args.server:
#         # get(data_path, parent=True)

#     # if run_args.server:
#         # get(route_data_path, parent=True)
#         # get(get_datadir() / training_setting["dataset"] / f"distance_matrix_bin{run_args.n_bins}.npy")
#         # if training_setting["network_type"] == "MTNet":
#             # get(get_datadir() / dataset_name / "raw", parent=True)

#     route_data_path = training_data_dir / "route_training_data.csv"
        
#     # with open(data_path / "params.json", "r") as f:
#         # data_setting = json.load(f)
#     # n_bins = int(np.sqrt(data_setting["n_locations"]) -2)
#     # assert n_bins == run_args.n_bins, "n_bins should be equal to the n_bins in the data"

#     # route_data_name = f"0_0_bin{n_bins}_seed{data_setting['seed']}"
#     # route_data_path = get_datadir() / training_setting["dataset"] / training_setting["data_name"] / route_data_name
    
#     # training_data_path = data_path
#     # if run_args.server:
#         # get(training_data_path, parent=True)
#     args = set_args(run_args)

#     eval_data_path = pathlib.Path(run_args.eval_data_dir) / "training_data.csv"
#     eval_route_data_path = pathlib.Path(run_args.eval_data_dir) / "route_training_data.csv"
#     dataset = construct_dataset(eval_data_path, eval_route_data_path, 5)
#     compute_auxiliary_information(dataset, model_dir, logger)

#     print(training_data_dir)
#     training_dataset = construct_dataset(training_data_dir / "training_data.csv", None, 5)
#     args.references = training_dataset.references
#     args.from_bin = training_dataset.n_bins
#     args.to_bin = dataset.n_bins
#     args.need_downsampling = (args.from_bin != args.to_bin)
#     if args.need_downsampling:
#         print("downsampling from", args.from_bin, "to", args.to_bin)
#         args.downsampling_dict = make_downsampling_dict(args.from_bin, args.to_bin)

#     args.save_dir = model_dir
#     (args.save_dir / f"imgs_trun{args.truncate}_{args.to_bin}").mkdir(exist_ok=True, parents=True)
    # make_first_order_test_data_loader(dataset, args.n_test_locations)
#     # make_second_order_test_data_loader(dataset, args.n_test_locations)
        
#     # sort according to i
#     model_paths = sorted(model_paths, key=lambda x: int(x.stem.split("_")[-1]))
#     print(model_paths)
#     for i, model_path in enumerate(model_paths):
#         if i % args.eval_interval != 0:
#             continue

#         if training_setting["network_type"] == "mtnet":
#             generator = MTNetGeneratorMock(model_path / "samples.txt", model_path / "samples_time.txt", training_setting["dataset"], n_bins)
#         elif training_setting["network_type"] == "privtrace":
#             with open(model_path / f"privtrace_generator.pickle", "rb") as f:
#                 generator = pickle.load(f)
#         elif training_setting["network_type"] == "clustering":
#             with open(model_path / f"generator.pickle", "rb") as f:
#                 generator = pickle.load(f)
#         elif training_setting["network_type"] in ["hrnet", "baseline"]:
#             meta_network, _ = construct_meta_network(training_setting["clustering"], training_setting["network_type"], training_dataset.n_locations, training_setting["memory_dim"], training_setting["memory_hidden_dim"], training_setting["location_embedding_dim"], training_setting["multilayer"], training_setting["consistent"], logger)
#             if hasattr(meta_network, "remove_class_to_query"):
#                 meta_network.remove_class_to_query()
#             generator, _ = construct_generator(training_dataset.n_locations, meta_network, training_setting["network_type"], training_setting["location_embedding_dim"], training_setting["n_split"], len(training_dataset.label_to_reference), training_setting["hidden_dim"], training_dataset.reference_to_label, logger)
#             logger.info(f"evaluate {model_path}!")
#             generator.load_state_dict(torch.load(model_path, map_location=device))
#             generator = generator.to(device)
#             print(generator, device)

#         args.name = model_path.name
#         results = run(generator, dataset, args)
#         print("save result to", args.save_dir / f"evaluated_{args.name}_trun{args.truncate}_{args.to_bin}.json")
#         with open(args.save_dir / f"evaluated_{args.name}_trun{args.truncate}_{args.to_bin}.json", "w") as f:
#             json.dump(results, f)
#         # if run_args.server:
#             # send(args.save_dir / f"evaluated_{args.name}_trun{args.truncate}_{args.to_bin}.json")
#             # send(args.save_dir / f"imgs_trun{args.truncate}_{args.to_bin}" / args.name, parent=True)
#         print(results)