from potentials_creation import read_from_pkl

class Node:
    def __init__(self,couple,potential):
        self.couple = couple
        self.potential = potential
        self.new_messages = {}
        self.old_messages = {} ### all_ones

    def pass_messages_to_neigh(self):
        return

    # def update_by_massage(self):
    #     return


class FactorGraph:
    def __init__(self, nodes_list, neighbors_dict):
        self.nodes = nodes_list
        self.neighbors_dict = neighbors_dict

    def belief_propagation(self):
        for node in self.nodes:
            node.pass_messages_to_neigh(self.neighbors_dict[node.couple])
        return


def find_neighbors(couple, couple_potentials,unobserved_size):
    neigh_list = []
    for couple_1 in couple_potentials:
        if couple_1[0] >= unobserved_size:
            break
        if (couple_1[0] == couple[0] or couple_1[0] == couple[1]
                or couple_1[1] == couple[0] or couple_1[1] == couple[1]):
            if couple_1 != couple:
                neigh_list.append(couple_1)
    return neigh_list




def main():
    all_single_potentials = read_from_pkl("all_data.pkl")
    unobserved_size = 250 ## to change
    couple_potentials = read_from_pkl("potentials_dict.pkl")
    nodes_list = []
    for couple in couple_potentials:
        new_node = Node(couple, couple_potentials[couple])
        nodes_list.append(new_node)
    neighbors_dict = {}
    for couple in couple_potentials:
        if couple[0] >= unobserved_size:
            break
        neighbors_dict[couple] = find_neighbors(couple, couple_potentials, unobserved_size)
    factor_graph = FactorGraph(nodes_list, neighbors_dict)
    print ("a")



if __name__ == '__main__':
    main()
