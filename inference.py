from potentials_creation import read_from_pkl,print_results
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def find_common(couple1, couple2):
    common_tuple = tuple(set(couple1).intersection(set(couple2)))
    if len(common_tuple) != 1:
        print("error in find common func")
        raise AttributeError
    common = common_tuple[0]
    place_of_common_in_first = [i for i in range(len(couple1)) if couple1[i] == common][0]
    place_of_common_in_second = [i for i in range(len(couple2)) if couple2[i] == common][0]
    return common, place_of_common_in_first,place_of_common_in_second


class Node:
    def __init__(self, couple, potential, unobserved_size):
        self.unobserved_size = unobserved_size
        self.couple = couple
        self.potential = potential
        self.messages_on_first = {}
        self.messages_on_second = {}
        self.final_belief = np.zeros((5, 5))
        self.final_return_values = {}

    def dis_of_massages(self):
        distance = 0
        counter = 0
        for m in self.messages_on_first:
            counter = counter + 1
            distance += np.mean(np.absolute(self.messages_on_first[m]['new'] - self.messages_on_first[m]['old']))
        for n in self.messages_on_second:
            counter = counter + 1
            distance += np.mean(np.absolute(self.messages_on_second[n]['new'] - self.messages_on_second[n]['old']))
        return distance,counter

    def update_old(self):
        for m in self.messages_on_first:
            self.messages_on_first[m]['old'] = self.messages_on_first[m]['new']
        for m in self.messages_on_second:
            self.messages_on_second[m]['old'] = self.messages_on_second[m]['new']

    def update_belief(self):
        delta_vector_first = np.ones(5)
        delta_vector_second = np.ones(5)
        for sender, massage in self.messages_on_second.items():
            delta_vector_first = np.multiply(delta_vector_first, massage['old'])
        delta_vector_first = delta_vector_first / delta_vector_first.sum(axis=0, keepdims=1)
        if self.couple[1] < self.unobserved_size:
            for sender, massage in self.messages_on_first.items():
                delta_vector_second = np.multiply(delta_vector_second, massage['old'])
        else:
            delta_vector_second = self.potential.to_numpy()[0, :]
        delta_vector_second = delta_vector_second / delta_vector_second.sum(axis=0, keepdims=1)
        raw_vector = np.array([delta_vector_second])
        column_vector = np.reshape(np.array([delta_vector_first]), (5, 1))
        d = np.dot(self.potential.to_numpy(), column_vector)
        self.final_belief = np.dot(d, raw_vector)
        self.final_belief = self.final_belief / self.final_belief.sum(keepdims=1)
        for i, house in enumerate(self.couple):
            self.final_return_values[house] = np.sum(self.final_belief, axis=1-i)
            self.final_return_values[house] = self.final_return_values[house]/self.final_belief.sum(keepdims=1)

    def initialize(self, neighbors_list):
        house1 = self.couple[0]
        house2 = self.couple[1]
        for neighbor in neighbors_list:
            if house1 == neighbor[0]:
                self.messages_on_first[neighbor[1]] = {}
                self.messages_on_first[neighbor[1]]['old'] = np.ones(5)
                self.messages_on_first[neighbor[1]]['new'] = np.ones(5)
            if house1 == neighbor[1]:
                self.messages_on_first[neighbor[0]] = {}
                self.messages_on_first[neighbor[0]]['old'] = np.ones(5)
                self.messages_on_first[neighbor[0]]['new'] = np.ones(5)
            if house2 == neighbor[0]:
                self.messages_on_second[neighbor[1]] = {}
                self.messages_on_second[neighbor[1]]['old'] = np.ones(5)
                self.messages_on_second[neighbor[1]]['new'] = np.ones(5)
            if house2 == neighbor[1]:
                self.messages_on_second[neighbor[0]] = {}
                self.messages_on_second[neighbor[0]]['old'] = np.ones(5)
                self.messages_on_second[neighbor[0]]['new'] = np.ones(5)
        return

    # def update_by_massage(self):
    #     return


class FactorGraph:

    def __init__(self, nodes_list, neighbors_dict):
        self.nodes = nodes_list
        self.neighbors_dict = neighbors_dict
        self.converge = False
        self.converge_value_list = []
        self.massage_count = 0

    def belief_propagation(self):
        self.initialize_massages()
        count = 0
        while not self.converge:
            print("BP iteration Number " + str(count))
            print(self.converge_value_list)
            self.send_messages()
            self.check_convergence()
            self.update_old()
            count += 1
        self.update_belief()

    def update_belief(self):
        for couple in self.nodes:
            self.nodes[couple].update_belief()

    def update_old(self):
        for couple in self.nodes:
            self.nodes[couple].update_old()

    def send_messages(self):
        for couple in self.nodes:
            self.pass_messages_to_neigh(self.nodes[couple])

    def inference_values(self, unobserved_size, single_potentials_before_belief):
        rows = []
        for i, house in single_potentials_before_belief.iterrows():
            house_to_dict = house.filter(items=[0,1,2,3,4])
            dict_to_fill = dict(house_to_dict)
            # dict_to_fill = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            if i >= unobserved_size:
                break
            for couple in self.nodes:
                if i in couple:
                    dict_to_fill_vec = self.nodes[couple].final_return_values[i][0]
                    for key, value in zip(dict_to_fill.keys(), dict_to_fill_vec):
                        dict_to_fill[key] = value
            list_of_values = list(dict_to_fill.values())
            if all(elem == list_of_values[0] for elem in list_of_values):
                dict_to_fill = dict(house_to_dict)
            rows.append(dict_to_fill)
        result_of_bp = pd.DataFrame(rows)
        return result_of_bp







    def pass_messages_to_neigh(self, sender_node):
        delta_vector_first = np.ones(5)
        delta_vector_second = np.ones(5)
        for sender, massage in sender_node.messages_on_second.items():
            delta_vector_first = np.multiply(delta_vector_first, massage['old'])
        delta_vector_first = np.dot(sender_node.potential.to_numpy(), delta_vector_first)
        delta_vector_first = delta_vector_first / delta_vector_first.sum(axis=0, keepdims=1)
        for sender, massage in sender_node.messages_on_first.items():
            delta_vector_second = np.multiply(delta_vector_second, massage['old'])
            delta_vector_second = delta_vector_second / delta_vector_second.sum(axis=0, keepdims=1)
        delta_vector_second = np.dot(sender_node.potential.to_numpy().transpose(), delta_vector_second)
        delta_vector_second = delta_vector_second / delta_vector_second.sum(axis=0, keepdims=1)
        neighbor_list = self.neighbors_dict[sender_node.couple]
        for neighbor in neighbor_list:
            common_point, first , second = find_common(sender_node.couple, neighbor)
            if first == 0:
                if second == 0:
                    self.nodes[neighbor].messages_on_first[sender_node.couple[1]]['new'] = delta_vector_first
                elif second == 1:
                    self.nodes[neighbor].messages_on_second[sender_node.couple[1]]['new'] = delta_vector_first
                else:
                    raise AttributeError
            elif first == 1:
                if second == 0:
                    self.nodes[neighbor].messages_on_first[sender_node.couple[0]]['new'] = delta_vector_second
                elif second == 1:
                    self.nodes[neighbor].messages_on_second[sender_node.couple[0]]['new'] = delta_vector_second
                else:
                    raise AttributeError
            else:
                raise AttributeError


            # if common_point == sender_node.couple[0]:
            #     place_of_common = 0
            # else:
            #     place_of_common = 1
            # if place_of_common == 0:

    def initialize_massages(self):
        for couple in self.nodes:
            self.nodes[couple].initialize(self.neighbors_dict[couple])

    def check_convergence(self):
        converge_value = 0
        msg_counts = 0
        for couple in self.nodes:
            conv_value, msg_count = self.nodes[couple].dis_of_massages()
            converge_value += conv_value
            msg_counts += msg_count
        self.converge_value_list.append(converge_value)
        self.massage_count = msg_counts
        if converge_value < 0.001:
            self.converge = True  # to be change by logical check


def find_neighbors(couple, couple_potentials, unobserved_size):
    neigh_list = []
    for couple_1 in couple_potentials:
        if couple_1[0] >= unobserved_size:
            break
        if (couple_1[0] == couple[0] or couple_1[0] == couple[1]
                or couple_1[1] == couple[0] or (couple_1[1] == couple[1] and couple[1] < unobserved_size)):
            if couple_1 != couple:
                neigh_list.append(couple_1)
    return neigh_list


def inference_graph(all_single_potentials,couple_potentials , unobserved_size):
    #all_single_potentials = read_from_pkl("single_potentials_all_nodes_lr_2.pkl")
    #couple_potentials = read_from_pkl("potentials_dict_all_nodes_lr_2.pkl")
    nodes_dict = {}
    neighbors_dict = {}
    for couple in couple_potentials:
        if couple[0] >= unobserved_size:
            break
        new_node = Node(couple, couple_potentials[couple], unobserved_size)
        nodes_dict[couple] = new_node
        neighbors_dict[couple] = find_neighbors(couple, couple_potentials, unobserved_size)
    factor_graph = FactorGraph(nodes_dict, neighbors_dict)
    factor_graph.belief_propagation()
    print(factor_graph)
    all_res_df = pd.DataFrame()
    result = factor_graph.inference_values(unobserved_size, all_single_potentials)
    y_pred_BP = result.idxmax(axis=1).values
    y_reg_pred = all_single_potentials[[0, 1, 2, 3, 4]].iloc[:unobserved_size].idxmax(axis=1).values
    all_res_df['BP'] = y_pred_BP
    all_res_df['reg'] = y_reg_pred
    all_res_df['true'] = all_single_potentials.iloc[:unobserved_size]['price']
    diffrence = all_res_df[all_res_df.BP!=all_res_df.reg]
    # print (len(diffrence))
    # print ("################# BP results ######################")
    # print_results(y_pred_BP, all_single_potentials.iloc[:unobserved_size]['price'])
    # print("################# reg results ######################")
    # print_results
    result_to_return = {
        'BP': accuracy_score(y_pred_BP,all_single_potentials.iloc[:unobserved_size]['price']),
        'regular': accuracy_score(y_reg_pred,all_single_potentials.iloc[:unobserved_size]['price'])
    }
    return result_to_return


if __name__ == '__main__':
    inference_graph()
