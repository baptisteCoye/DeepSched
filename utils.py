import numpy


def graph_to_matrix(dag):
    matrix = numpy.zeros((dag.size, dag.size))
    for edge in dag.edges:
        matrix[edge[0], edge[1]] = 1
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
                if matrix[i][j] == 1:
                    matrix[i][j] = -1
                if matrix[j][i] == 1:
                    matrix[j][i] = -1
    return matrix


def not_finished(dag):
    return len(dag.solution) != 0


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
        if(value > 0.01):
            raw_solution[index] = 0
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


