import numpy


def fully_labeled(label_array, size):
    res = 0
    for i in range(size):
        if label_array[i] == 0:
            res += 1
    return res == 0


def is_labeled(label_array, node):
    return label_array[node] != 0


def all_successor_labeled(label_array, node, edges, size):
    res = 0
    for j in range(size):
        if (node, j) in edges and not is_labeled(label_array, j):
            res += 1
    return res == 0


def lexicographic_value(label_array, node, edges, size):
    tmp = 0
    successor_value = list()
    for j in range(size):
        if (node, j) in edges:
            successor_value.append(label_array[j])
    successor_value.sort()
    successor_value.reverse()
    return successor_value


def sort_lexicographic(lexicographic_list, index_list):
        solution = list()
        index = list()
        while len(lexicographic_list) != 0:
            # print(lexicographic_list)
            temporary_min = lexicographic_list[0]
            temporary_index = index_list[0]
            for i in range(len(lexicographic_list)):
                tmp = lexicographic_min(temporary_min,lexicographic_list[i])
                if(tmp != temporary_min):
                    temporary_min = tmp
                    temporary_index = index_list[i]
            solution.append(temporary_min)
            index.append(temporary_index)
            lexicographic_list.remove(temporary_min)
            index_list.remove(temporary_index)
        return (solution, index)


def lexicographic_min(first, second):
    # print(first)
    # print(second)
    # print("\n")
    for i in range(max(len(first), len(second))):
        if(i > len(first)-1):
            return second
        elif(i > len(second)-1):
            return first
        else:
            if first[i] < second[i]:
                return first
            elif second[i] < first[i]:
                return second
    return first


def distance_pit(dag):
    label_array = numpy.zeros(dag.size)
    distance_to_pit = 0
    while not fully_labeled(label_array, dag.size):
        list_to_index = list()
        for i in range(dag.size):
            if all_successor_labeled(label_array, i, dag.edges, dag.size) and not is_labeled(label_array, i):
                list_to_index.append(i)
        for j in list_to_index:
            label_array[j] = distance_to_pit
        distance_to_pit += 1
    return label_array


def get_successors(node, edges, size):
    successors = list()
    for i in range(size):
        if (node, i) in edges:
            successors.append(i)
    return successors


def max_no_zeros(first_value, second_value):
    if first_value == 0:
        return second_value
    elif second_value == 0:
        return first_value
    else:
        if first_value > second_value:
            return first_value
        else:
            return second_value


def get_max_distance(successors, matrix, size):
    distance_vector = numpy.zeros(size)
    for j in range(len(successors)):
        for i in range(size):
            distance_vector[i] = max_no_zeros(distance_vector[i], matrix[successors[j]][i])
    return distance_vector


def update_distance_vector(distance_vector, successors):
    for i in range(len(distance_vector)):
        if distance_vector[i] != 0:
            distance_vector[i] += 1
    for j in range(len(successors)):
        distance_vector[successors[j]] = 1
    return distance_vector

def update_distance_matrix(matrix, node, distance_vector):
    for i in range(len(distance_vector)):
        matrix[node][i] = distance_vector[i]

def distance_matrix(dag):
    label_array = numpy.zeros(dag.size)
    matrix = numpy.zeros((dag.size, dag.size))
    while not fully_labeled(label_array, dag.size):
        for i in range(dag.size):
            if all_successor_labeled(label_array, i, dag.edges, dag.size) and not is_labeled(label_array, i):
                successors = get_successors(i, dag.edges, dag.size)
                distance_vector = get_max_distance(successors, matrix, dag.size)
                update_distance_vector(distance_vector, successors)
                update_distance_matrix(matrix, i, distance_vector)
                label_array[i] = 1
    return matrix

