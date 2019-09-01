import inference
import potentials_creation
import mrf
import math
import json
import matplotlib.pyplot as plt
import numpy as np

BASIC_HYPER_PARAMS = {
    'MAX_DIST': 0.0175,
    'TEST_SPLIT': 0.1,
    'NUMBER_OF_LABELS': 5,
    'TOTAL_SIZE': 2000
}


def analyze_number_of_edges():
    max_dist_values = [0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.0275, 0.03]
    test_split = BASIC_HYPER_PARAMS['TEST_SPLIT']
    total_size = BASIC_HYPER_PARAMS['TOTAL_SIZE']
    all_data = potentials_creation.create_data(test_split, total_size)
    unobserved_size = math.ceil(test_split * total_size)
    results = dict()
    results['max_dist'] = []
    results['number_of_edges'] = []
    results['BP_acc'] = []
    results['reg_acc'] = []
    for dist in max_dist_values:
        neigh_dict = mrf.get_neighbors_dict(all_data, unobserved_size, dist)
        results['max_dist'].append(dist)
        results['number_of_edges'].append(len(neigh_dict))
        potentials = potentials_creation.create_pairwise_potentials(neigh_dict, all_data)
        result_dict = inference.inference_graph(all_data, potentials,unobserved_size)
        results['BP_acc'].append(result_dict['BP'])
        results['reg_acc'].append(result_dict['regular'])
        print(results)
    with open('results_of_number_of_edges_svm.json', 'w') as file:
        json.dump(results, file)
    print(results)


def make_dist_graphs():
    with open('results_of_number_of_edges_svm.json', 'r') as file:
        results = json.load(file)
    plt.plot(results["max_dist"], results["number_of_edges"])
    plt.xlabel('max distance between houses')
    plt.ylabel('number of edges')
    plt.title('Number of edges')
    plt.savefig('Number_of_edges_svm.png')
    plt.clf()
    plt.plot(results["number_of_edges"], np.array(results["BP_acc"])*100, label="BP_acc" ,scalex=[])
    plt.plot(results["number_of_edges"], np.array(results["reg_acc"])*100, label="reg_acc")
    plt.xlabel('number of edges')
    plt.ylabel('accuracy')
    plt.title('accuracy comparison')
    plt.legend(loc='lower left')
    plt.savefig('Number_of_edges_accuracy_comparison_svm.png')


def analyze_observed_size():
    test_splits = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
    max_dist = BASIC_HYPER_PARAMS['MAX_DIST']
    total_size = BASIC_HYPER_PARAMS['TOTAL_SIZE']
    results = dict()
    results['un_observed_split'] = []
    results['number_of_edges'] = []
    results['BP_acc'] = []
    results['reg_acc'] = []
    for test_split in test_splits:
        all_data = potentials_creation.create_data(test_split, total_size)
        unobserved_size = math.ceil(test_split * total_size)
        neigh_dict = mrf.get_neighbors_dict(all_data, unobserved_size, max_dist)
        results['un_observed_split'].append(test_split)
        results['number_of_edges'].append(len(neigh_dict))
        potentials = potentials_creation.create_pairwise_potentials(neigh_dict, all_data)
        result_dict = inference.inference_graph(all_data, potentials, unobserved_size)
        results['BP_acc'].append(result_dict['BP'])
        results['reg_acc'].append(result_dict['regular'])
        print(results)
    with open('results_of_different_test_splits_3_svm.json', 'w') as file:
        json.dump(results, file)
    print(results)


def make_test_split_graphs():
    with open('results_of_different_test_splits_3_svm.json', 'r') as file:
        results = json.load(file)
    plt.plot(results["un_observed_split"], results["number_of_edges"])
    plt.xlabel('unobserved ratio')
    plt.ylabel('number of edges')
    plt.title('Number of edges for unobserved ratio')
    plt.savefig('Number_of_edges_for_test_split_svm.png')
    plt.clf()
    plt.plot(results["un_observed_split"], results["BP_acc"], label="BP_acc")
    plt.plot(results["un_observed_split"], results["reg_acc"], label="reg_acc")
    plt.xlabel('unobserved ratio')
    plt.ylabel('accuracy')
    plt.title('accuracy comparison')
    plt.legend(loc='lower left')
    plt.savefig('unobserved_ratio_accuracy_comparison_svm.png')


def run_test():
    test_split = BASIC_HYPER_PARAMS['TEST_SPLIT']
    total_size = BASIC_HYPER_PARAMS['TOTAL_SIZE']
    max_dist = BASIC_HYPER_PARAMS['MAX_DIST']
    all_data = potentials_creation.create_data(test_split, total_size)
    #all_data = potentials_creation.read_from_pkl('new_data.pkl')
    unobserved_size = math.ceil(test_split * total_size)
    #neigh_dict = potentials_creation.read_from_pkl('new_neighbor_dict.pkl')
    edges_probabilities = mrf.get_edges_probs(all_data, unobserved_size, max_dist)
    potentials_creation.write_to_pkl(edges_probabilities, 'edges_probabilities_sim.pkl')
    neigh_dict = mrf.get_neighbors_dict(all_data, unobserved_size, max_dist)
    potentials_creation.write_to_pkl(all_data, 'new_data_sim.pkl')
    potentials_creation.write_to_pkl(neigh_dict, 'new_neighbor_dict_sim.pkl')
    potentials = potentials_creation.create_pairwise_potentials(neigh_dict, all_data, edges_probabilities)
    result_dict = inference.inference_graph(all_data, potentials, unobserved_size)
    print(result_dict)

def main():
    run_test()


if __name__ == '__main__':
    main()





