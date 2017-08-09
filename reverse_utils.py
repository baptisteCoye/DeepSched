import numpy
import random

def graph_to_matrix(dag):
    matrix = numpy.zeros((dag.size, dag.size)) + 1
    for edge in dag.edges:
        matrix[edge[0], edge[1]] = 0
    return matrix


def transpose_matrix(matrix, size_x, size_y):
    new_matrix = numpy.zeros((size_y, size_x))
    for i in range(size_y):
        for j in range(size_x):
            new_matrix[i, j] = matrix[j, i]
    return new_matrix


def update_matrix(matrix, real_solution, size):
    res = 0
    for i in range(size):
        if real_solution[i] == 1:
            for j in range(size):
                if matrix[i][j] == 0:
                    matrix[i][j] = 1
                if matrix[j][i] == 0:
                    matrix[j][i] = 1
    return matrix


def not_finished(dag):
    return len(dag.edges) != 0


def is_schedulable(dag, task):
    for i in range(dag.size):
        if (i, task) in dag.edges:
            return False
    return True


def get_schedulable_tasks(dag, already_scheduled):
    schedulable_tasks = numpy.zeros(dag.size)
    for i in range(dag.size):
        res = 0
        for j in range(dag.size):
            if (j, i) in dag.edges:
                res += 1
        if res == 0:
            schedulable_tasks[i] = 1
    return schedulable_tasks - already_scheduled


def find_solution(dag):
    temporary_solution = numpy.zeros(dag.size)
    first = dag.solution[0]
    second = -1
    dag.solution.remove(dag.solution[0])
    for i in range(len(dag.solution)):
        if is_schedulable(dag, dag.solution[i]):
            second = dag.solution[i]
            dag.solution.remove(dag.solution[i])
            break
    temporary_solution[first] = 1
    if second != -1:
        temporary_solution[second] = 1
    return temporary_solution


def update_solution(dag, solution):
    for i in range(len(solution)):
        if solution[i] == 1:
            dag.solution.remove(i)


def update_dag(dag, solution):
    for i in range(dag.size):
        if solution[i] == 1:
          for j in range(dag.size):
            if (i, j) in dag.edges:
                dag.edges.remove((i, j))


def post_process(raw_solution):
    processed_solution = numpy.zeros(len(raw_solution))
    for i in range(2):
        index_and_value = get_max_index(raw_solution)
        index = index_and_value[0]
        value = index_and_value[1]
        if(value > 0.1):
            raw_solution[index] = 0
            processed_solution[index] = 1
    return processed_solution


def post_process_real(solution):
    processed_solution = numpy.zeros(len(solution))
    for i in range(2):
        index_and_value = get_max_index(solution)
        index = index_and_value[0]
        value = index_and_value[1]
        if (value > 0):
            solution[index] = 0
            processed_solution[index] = 1
    return processed_solution


def get_max_index(vector):
    tmp = vector[0]
    index = 0
    for i in range(len(vector)):
        if vector[i] > tmp:
            index = i
            tmp = vector[i]
    return (index, tmp)


def get_max_index_list(vector):
    tmp_value = vector[0]
    index = list()
    for i in range(len(vector)):
        if vector[i] > tmp_value:
            tmp_value = vector[i]
            index = list()
            index.append(i)
        elif vector[i] == tmp_value:
            index.append(i)
    return index


def probabilist_solution(schedulable, dag_solution):
    mask = schedulable*dag_solution
    prob_sol = numpy.zeros(len(dag_solution))
    index = get_max_index_list(mask)

    if empty(mask):
        return prob_sol

    if len(index) > 1:
        for i in range(len(index)):
            prob_sol[index[i]] = 2/len(index)
        return prob_sol
    elif len(index) == 1:
        prob_sol[index[0]] = 1
        mask[index[0]] = 0
        second_index = get_max_index_list(mask)
        if not empty(mask):
            for i in range(len(second_index)):
                prob_sol[second_index[i]] = 1/len(second_index)
        return prob_sol


def empty(vector):
    for i in range(len(vector)):
        if vector[i] != 0:
            return False
    return True


def negative_value(vector):
    for i in range(len(vector)):
        if vector[i] < 0:
            return True
    return False


def process_solution(prob_solution):
    real_solution = numpy.zeros(len(prob_solution))
    if empty(prob_solution):
        return real_solution
    index_list = get_max_index_list(prob_solution)
    if len(index_list) == 3:
        real_solution[index_list[0]] = 1
        real_solution[index_list[1]] = 1
        return real_solution
    elif len(index_list) > 3:
        random.shuffle(index_list)
        real_solution[index_list[0]] = 1
        real_solution[index_list[1]] = 1
        return real_solution
    elif len(index_list) == 2:
        real_solution[index_list[0]] = 1

    elif len(index_list) == 1:
        real_solution[index_list[0]] = 1
        prob_solution[index_list[0]] = 0
        if empty(prob_solution):
            return real_solution
        else:
            index_list2 = get_max_index_list(prob_solution)
            random.shuffle(index_list2)
            real_solution[index_list2[0]] = 1
            return real_solution
