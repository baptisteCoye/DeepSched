from DAG import *
from reverse_utils import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *


graph_size = 20
nb_epochs = 200
nb_test = 100


##########################################################################################
#                            Deep Neural Network Definition                              #
##########################################################################################

model=Sequential()
model.add(Dense(512,activation='relu', input_dim=graph_size+2))
model.add(PReLU())
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


for i in range(nb_epochs):
    #############################################
    #               Initialisation              #
    #############################################
    graph = DAG(graph_size)
    graph.solution = distance_pit(graph)
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
        print(schedulable_list)
        # print(solution)
        # training_matrix = transpose_matrix(matrix, graph_size, graph_size)
        preprocessed_matrix = numpy.column_stack([matrix, schedulable_list, already_scheduled])

        # training_matrix = transpose_matrix(preprocessed_matrix, graph_size, graph_size+2)
        ##########################################
        #       Create Value to Feed Model       #
        ##########################################
        X_train = preprocessed_matrix
        Y_train = solution
        print(model.predict(X_train))
        model.fit(x=X_train, y=Y_train, batch_size=1, nb_epoch=1)

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
count_fail = 0
count_success = 0
count_crash = 0
count_best = 0

for i in range(nb_test):
    graph = DAG(graph_size)
    # graph.renumerotation()
    graph.solution = distance_pit(graph)
    graph_test = DAG(graph_size)
    graph_test.edges = graph.edges.copy()
    graph_test.solution = graph.solution.copy()
    matrix = graph_to_matrix(graph)
    matrix_test = graph_to_matrix(graph_test)
    already_scheduled = numpy.zeros(graph_size)
    already_scheduled_test = numpy.zeros(graph_size)
    execution_time = 0
    execution_time_test = 0

    while not_finished(graph_test):

        schedulable_list_test = get_schedulable_tasks(graph_test, already_scheduled_test)
        print(schedulable_list_test)
        preprocessed_matrix = numpy.column_stack([matrix_test, schedulable_list_test, already_scheduled_test])
        X_train = preprocessed_matrix
        prediction = model.predict(X_train)
        real_solution = post_process(prediction)
        update_dag(graph_test, real_solution)
        already_scheduled_test = already_scheduled_test + real_solution
        execution_time_test += 1

        if negative_value(schedulable_list_test - real_solution):
            count_crash += 1

        if empty(real_solution):
            print(prediction)
        print(real_solution)

    print("\n")
    while not_finished(graph):
        schedulable_list = get_schedulable_tasks(graph, already_scheduled)
        print(schedulable_list)
        solution = probabilist_solution(schedulable_list, graph.solution)
        chosen_solution = post_process_real(solution)
        # update_matrix(matrix, chosen_solution, graph_size)
        update_dag(graph, chosen_solution)
        # print(solution)
        already_scheduled = already_scheduled + chosen_solution
        execution_time += 1
        print(chosen_solution)

    if execution_time_test > execution_time:
        print(execution_time_test - execution_time)
        count_fail += 1
    elif execution_time_test < execution_time:
        count_best += 1
    else:
        print(execution_time_test - execution_time)
        print('\n')
        count_success += 1

print(count_crash)
print(count_fail)
print(count_success)
print(count_best)