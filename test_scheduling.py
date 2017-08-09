
from DAG import *
from utils_test import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import time
import random


def create_broadcast(storey_number):
    broadcast_graph = DAG((storey_number+1)*4+2)
    random_list = list()
    for i in range(storey_number):
        random_list.append(random.randint(0, 3))
    broadcast_graph.edges = set()
    for i in range(storey_number):
        for j in range(4):
            broadcast_graph.edges.add((4*i+1+random_list[i], 4*(i+1)+j+1))
    for i in range(4):
        broadcast_graph.edges.add((0, i+1))
        broadcast_graph.edges.add((storey_number*4+1+i, 4*(storey_number+1)+1))
    for i in range(storey_number*4):
        broadcast_graph.edges.add((i+1, i+5))
    return broadcast_graph

nb_storey = 6
graph_size = (nb_storey+1)*4+2
nb_epochs = 300
nb_steps = 40
nb_test = 5000
nb_charts = 10
data = open("data.txt", "w")
execution = open("execution_time.txt", "w")
worse_scheduling_time = 2*(nb_storey+2)-1


##########################################################################################
#                            Deep Neural Network Definition                              #
##########################################################################################

model=Sequential()
model.add(Dense(512,activation='relu', input_dim=graph_size+2))
model.add(PReLU())
model.add(MaxPooling1D(pool_size=2, strides=None, padding='valid'))
model.add(Dense(512,activation='linear'))
model.add(PReLU())
model.add(Dense(512,activation='relu'))
model.add(PReLU())
model.add(Dense(256, activation='linear'))
model.add(PReLU())
model.add(Dense(256, activation='relu'))
model.add(PReLU())
model.add(Dense(256, activation='linear'))
model.add(PReLU())
model.add(Dense(128, activation='relu'))
model.add(PReLU())
model.add(Dense(128, activation='linear'))
model.add(PReLU())
model.add(Dense(128, activation='relu'))
model.add(PReLU())
model.add(Dense(64,activation='relu'))
model.add(PReLU())
model.add(Dense(64, activation='linear'))
model.add(PReLU())
model.add(Dense(64, activation='relu'))
model.add(PReLU())
model.add(Dense(32, activation='linear'))
model.add(PReLU())
model.add(Dense(32, activation='relu'))
model.add(PReLU())
model.add(Dense(32, activation='linear'))
model.add(PReLU())
model.add(Dense(16, activation='relu'))
model.add(PReLU())
model.add(Dense(16, activation='linear'))
model.add(PReLU())
model.add(Dense(graph_size, activation='relu'))
model.add(Dense(graph_size, activation='linear'))
model.add(Dense(1, activation='softplus'))
model.compile(Adagrad(lr=1e-3, decay=1e-8), loss="mean_squared_error")


##########################################################################################
#                                    Training Model                                      #
##########################################################################################
start = time.time()
for j in range(nb_steps):
    for i in range(nb_epochs):
        #############################################
        #               Initialisation              #
        #############################################
        graph = DAG(graph_size)
        graph.solution = distance_pit(graph)
        # print(graph.edges)
        # graph.renumerotation()
        matrix = graph_to_matrix(graph)
        already_scheduled = numpy.zeros(graph_size)
        while not_finished(graph):

            #########################################
            #               Preprocessing           #
            #########################################
            schedulable_list = get_schedulable_tasks(graph, already_scheduled)
            solution = probabilist_solution(schedulable_list, graph.solution)
            # print(matrix)
            # print(schedulable_list)
            # print(solution)
            # training_matrix = transpose_matrix(matrix, graph_size, graph_size)
            preprocessed_matrix = numpy.column_stack([matrix, schedulable_list, already_scheduled])

            # training_matrix = transpose_matrix(preprocessed_matrix, graph_size, graph_size+2)
            ##########################################
            #       Create Value to Feed Model       #
            ##########################################
            X_train = preprocessed_matrix
            Y_train = solution
            X_train = X_train.reshape(1 + X_train.shape[0])
            # print(model.predict(X_train))
            model.fit(x=X_train, y=Y_train, batch_size=100, nb_epoch=1)

            ##########################################
            #           Update for next Turn         #
            ##########################################
            chosen_solution = post_process_real(solution)
            # update_matrix(matrix, chosen_solution, graph_size)
            update_dag(graph, chosen_solution)
            already_scheduled = already_scheduled + chosen_solution
    ##########################################################################################
    #                                     Testing Model                                      #
    ##########################################################################################
    end = time.time()
    print(end - start)

    count_fail = 0
    count_success_1 = 0
    count_success_2 = 0
    count_success_3 = 0
    count_crash = 0
    count_best = 0
    count_fail_test = 0
    count_success_1_test = 0
    count_success_2_test = 0
    count_success_3_test = 0
    count_crash_test = 0
    count_best_test = 0

    total_ratio = 0

    #
    # for i in range(nb_test):
    #     graph = DAG(graph_size)
    #     # graph.renumerotation()
    #     graph.solution = distance_pit(graph)
    #     graph_test = DAG(graph_size)
    #     graph_test.edges = graph.edges.copy()
    #     graph_test.solution = graph.solution.copy()
    #     matrix = graph_to_matrix(graph)
    #     matrix_test = graph_to_matrix(graph_test)
    #     already_scheduled = numpy.zeros(graph_size)
    #     already_scheduled_test = numpy.zeros(graph_size)
    #     execution_time = 0
    #     execution_time_test = 0
    #
    #     while not_finished(graph_test):
    #
    #         schedulable_list_test = get_schedulable_tasks(graph_test, already_scheduled_test)
    #         # print(schedulable_list_test)
    #         preprocessed_matrix = numpy.column_stack([matrix_test, schedulable_list_test, already_scheduled_test])
    #         X_train = preprocessed_matrix
    #         prediction = model.predict(X_train)
    #         real_solution = post_process(prediction)
    #         update_dag(graph_test, real_solution)
    #         already_scheduled_test = already_scheduled_test + real_solution
    #         execution_time_test += 1
    #
    #         if negative_value(schedulable_list_test - real_solution):
    #             count_crash += 1
    #             execution_time_test = 1000
    #             break
    #         if empty(real_solution):
    #             # print(prediction)
    #             count_crash += 1
    #             execution_time_test = 1000
    #             break
    #         # print(real_solution)
    #
    #     # print("\n")
    #     while not_finished(graph):
    #         schedulable_list = get_schedulable_tasks(graph, already_scheduled)
    #         # print(schedulable_list)
    #         solution = probabilist_solution(schedulable_list, graph.solution)
    #         chosen_solution = post_process_real(solution)
    #         # update_matrix(matrix, chosen_solution, graph_size)
    #         update_dag(graph, chosen_solution)
    #         # print(solution)
    #         already_scheduled = already_scheduled + chosen_solution
    #         execution_time += 1
    #         # print(chosen_solution)
    #
    #     if execution_time_test > execution_time:
    #         # print(execution_time_test - execution_time)
    #         count_fail += 1
    #     elif execution_time_test < execution_time:
    #         count_best += 1
    #     else:
    #         # print(execution_time_test - execution_time)
    #         # print('\n')
    #         count_success += 1
    #

    for i in range(nb_test):

        graph_test = create_broadcast(nb_storey)
        graph_test.solution = distance_pit(graph_test)
        graph = DAG(graph_size)
        graph.edges = graph_test.edges.copy()
        graph.solution = graph_test.solution.copy()
        already_scheduled = numpy.zeros(graph_size)
        already_scheduled_test = numpy.zeros(graph_size)
        matrix = graph_to_matrix(graph)
        matrix_test = graph_to_matrix(graph_test)
        execution_time = 0
        execution_time_test = 0
        while not_finished(graph_test):

            schedulable_list_test = get_schedulable_tasks(graph_test, already_scheduled_test)
            # print(schedulable_list_test)
            preprocessed_matrix = numpy.column_stack([matrix_test, schedulable_list_test, already_scheduled_test])
            X_train = preprocessed_matrix
            prediction = model.predict(X_train)
            real_solution = post_process(prediction)
            update_dag(graph_test, real_solution)
            already_scheduled_test = already_scheduled_test + real_solution
            execution_time_test += 1
            # print(real_solution)

            if negative_value(schedulable_list_test - real_solution):
                count_crash += 1
                execution_time_test = 1000
                break
            if empty(real_solution):
                # print(prediction)
                count_crash += 1
                execution_time_test = 1000
                break
                # print(real_solution)
        print("\n")

        while not_finished(graph):
            schedulable_list = get_schedulable_tasks(graph, already_scheduled)
            solution = probabilist_solution(schedulable_list, graph.solution)
            chosen_solution = post_process_real(solution)
            update_dag(graph, chosen_solution)
            already_scheduled = already_scheduled + chosen_solution
            execution_time += 1
        print(worse_scheduling_time)
        print(execution_time_test)
        if execution_time_test == worse_scheduling_time:
            count_fail_test += 1
        if execution_time_test == worse_scheduling_time - 1:
            count_success_1_test += 1
        if execution_time_test == worse_scheduling_time - 2:
            count_success_2_test += 1
        if execution_time_test == worse_scheduling_time - 3:
            count_success_3_test += 1
        if execution_time_test < worse_scheduling_time - 3:
            count_best_test += 1
        print(execution_time)

        if execution_time == worse_scheduling_time:
            count_fail += 1
        if execution_time == worse_scheduling_time - 1:
            count_success_1 += 1
        if execution_time == worse_scheduling_time - 2:
            count_success_2 += 1
        if execution_time == worse_scheduling_time - 3:
            count_success_3 += 1
        if execution_time < worse_scheduling_time - 3:
            count_best += 1

        execution.write(str(execution_time_test))
        execution.write("\n")
        execution.write(str(execution_time))
        execution.write("\n")
        total_ratio += execution_time_test - 11
        if execution_time < execution_time_test:
            execution.write("BAD")
        elif execution_time == execution_time_test:
            execution.write("OK")
        else:
            execution.write("GOOD")
        execution.write("\n")
        execution.write("\n")




    data.write(str(j*nb_epochs))
    data.write("\n")
    data.write(str(count_crash))
    data.write("  ")
    data.write(str(count_crash_test))
    data.write("\n")
    data.write(str(count_success_1))
    data.write("  ")
    data.write(str(count_success_1_test))
    data.write("\n")
    data.write(str(count_success_2))
    data.write("  ")
    data.write(str(count_success_2_test))
    data.write("\n")
    data.write(str(count_success_3))
    data.write("  ")
    data.write(str(count_success_3_test))
    data.write("\n")
    data.write(str(count_best))
    data.write("  ")
    data.write(str(count_best_test))
    data.write("\n")
    data.write(str(total_ratio))
    data.write("\n")
    print(total_ratio)
    data.write("\n")
    data.write("\n")
