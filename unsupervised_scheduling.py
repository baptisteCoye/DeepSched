
from DAG import *
from utils_unsupervised import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import time
import random


def create_broadcast(storey_number):
    broadcast_graph = DAG((storey_number+1)*4+2,2)
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
nb_epochs = 900
nb_epochs_master = 1000
nb_steps = 40
nb_test = 5000
nb_charts = 10
data = open("data.txt_unsupervised", "w")
execution = open("execution_time_unsupervised.txt", "w")
worse_scheduling_time = 2*(nb_storey+2)-1
nb_unsupervised_training = 10000

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

model_master = Sequential()
model_master.add(Dense(512,activation='relu', input_dim=graph_size+2))
model_master.add(PReLU())
model_master.add(Dense(512,activation='linear'))
model_master.add(PReLU())
model_master.add(Dense(512,activation='relu'))
model_master.add(PReLU())
model_master.add(Dense(256, activation='linear'))
model_master.add(PReLU())
model_master.add(Dense(256, activation='relu'))
model_master.add(PReLU())
model_master.add(Dense(256, activation='linear'))
model_master.add(PReLU())
model_master.add(Dense(128, activation='relu'))
model_master.add(PReLU())
model_master.add(Dense(128, activation='linear'))
model_master.add(PReLU())
model_master.add(Dense(128, activation='relu'))
model_master.add(PReLU())
model_master.add(Dense(64,activation='relu'))
model_master.add(PReLU())
model_master.add(Dense(64, activation='linear'))
model_master.add(PReLU())
model_master.add(Dense(64, activation='relu'))
model_master.add(PReLU())
model_master.add(Dense(32, activation='linear'))
model_master.add(PReLU())
model_master.add(Dense(32, activation='relu'))
model_master.add(PReLU())
model_master.add(Dense(32, activation='linear'))
model_master.add(PReLU())
model_master.add(Dense(16, activation='relu'))
model_master.add(PReLU())
model_master.add(Dense(16, activation='linear'))
model_master.add(PReLU())
model_master.add(Dense(graph_size, activation='relu'))
model_master.add(Dense(graph_size, activation='linear'))
model_master.add(Dense(1, activation='softplus'))
model_master.compile(Adagrad(lr=1e-3, decay=1e-8), loss="mean_squared_error")

##########################################################################################
#                             Supervised Training                                        #
##########################################################################################

for i in range(nb_epochs):
    #############################################
    #               Initialisation              #
    #############################################
    graph = DAG(graph_size, 2)
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
        preprocessed_matrix = numpy.column_stack([matrix, schedulable_list, already_scheduled])
        ##########################################
        #       Create Value to Feed Model       #
        ##########################################
        X_train = preprocessed_matrix
        Y_train = solution
        # print(model.predict(X_train))
        model.fit(x=X_train, y=Y_train, batch_size=100, nb_epoch=1)

        ##########################################
        #           Update for next Turn         #
        ##########################################
        chosen_solution = post_process_real(solution)
        update_dag(graph, chosen_solution)
        already_scheduled = already_scheduled + chosen_solution

##########################################################################################
#                          Supervised Training for master                                #
##########################################################################################
print("TRAINING MASTER \n \n \n \n \n \n")
for i in range(nb_epochs_master):
        #############################################
        #               Initialisation              #
        #############################################
        graph = DAG(graph_size, 2)
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
            preprocessed_matrix = numpy.column_stack([matrix, schedulable_list, already_scheduled])
            ##########################################
            #       Create Value to Feed Model       #
            ##########################################
            X_train = preprocessed_matrix
            Y_train = solution
            # print(model.predict(X_train))
            model_master.fit(x=X_train, y=Y_train, batch_size=100, nb_epoch=1)

            ##########################################
            #           Update for next Turn         #
            ##########################################
            chosen_solution = post_process_real(solution)
            update_dag(graph, chosen_solution)
            already_scheduled = already_scheduled + chosen_solution

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

total_ratio_before = 0

for i in range(nb_test):

    graph_test = create_broadcast(nb_storey)
    graph_test.solution = distance_pit(graph_test)
    graph = DAG(graph_size, 2)
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
        print(schedulable_list_test)
        X_train = preprocessed_matrix
        prediction = model.predict(X_train)
        print(prediction)
        print("\n")
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
    total_ratio_before += execution_time_test - 11
    if execution_time < execution_time_test:
        execution.write("BAD")
    elif execution_time == execution_time_test:
        execution.write("OK")
    else:
        execution.write("GOOD")
    execution.write("\n")
    execution.write("\n")

##########################################################################################
#                           Unsupervised Training                                        #
##########################################################################################
print("UNSUPERVISED TRAINING \n \n \n \n \n \n BRACE YOURSELVES")
for j in range(nb_unsupervised_training):
    print("EPOCHS NUMBER:")
    print(j)
    graph = create_broadcast(nb_storey)
    graph_master = DAG(graph_size, 2)
    graph_trainer = DAG(graph_size, 2)
    graph_trainer.edges = graph.edges.copy()
    graph_master.edges = graph.edges.copy()
    already_scheduled = numpy.zeros(graph_size)
    already_scheduled_trainer = numpy.zeros(graph_size)
    already_scheduled_master = numpy.zeros(graph_size)
    matrix = graph_to_matrix(graph)
    matrix_trainer = graph_to_matrix(graph_trainer)
    matrix_master = graph_to_matrix(graph_master)
    execution_time = 0
    execution_time_master = 0
    count_master_win = 0
    master_value = list()
    master_scheduling = list()
    student_value = list()
    student_scheduling = list()
    while not_finished(graph):
        schedulable_list = get_schedulable_tasks(graph, already_scheduled)
        preprocessed_matrix = numpy.column_stack([matrix, schedulable_list, already_scheduled])
        X_train = preprocessed_matrix
        prediction = model.predict(X_train)

        student_value.append(prediction.copy())
        real_solution = post_process(prediction)
        update_dag(graph, real_solution)
        already_scheduled = already_scheduled + real_solution
        student_scheduling.append(real_solution)
        execution_time += 1
        if negative_value(schedulable_list - real_solution):
            execution_time = 1000
            break
        if empty(real_solution):
            execution_time = 1000
            break

    while not_finished(graph_master):
        schedulable_list = get_schedulable_tasks(graph_master, already_scheduled_master)
        preprocessed_matrix= numpy.column_stack([matrix_master, schedulable_list, already_scheduled_master])
        X_train = preprocessed_matrix
        prediction = model_master.predict(X_train)
        master_value.append(prediction.copy())
        real_solution = post_process(prediction)
        update_dag(graph_master, real_solution)
        already_scheduled_master = already_scheduled_master + real_solution
        execution_time_master += 1
        master_scheduling.append(real_solution)
        if negative_value(schedulable_list - real_solution):
            execution_time_master = 1000
            break
        if empty(real_solution):
            execution_time_master = 1000
            break

    print("\n")
    print(execution_time_master)
    print(execution_time)
    print("\n")

    if execution_time_master < execution_time < 1000 and execution_time_master == 11:
        schedule_step = 0
        while not_finished(graph_trainer):
            schedulable_list = get_schedulable_tasks(graph_trainer, already_scheduled_trainer)
            preprocessed_matrix = numpy.column_stack([matrix_trainer, schedulable_list, already_scheduled_trainer])
            X_train = preprocessed_matrix
            Y_train = probabilist_solution_real(schedulable_list, master_value[schedule_step])
            model.fit(x=X_train, y=Y_train, batch_size=10, nb_epoch=1)
            real_solution = master_scheduling[schedule_step]
            update_dag(graph_trainer, real_solution)
            already_scheduled_trainer = already_scheduled_trainer + real_solution
            schedule_step += 1
    elif 1000 > execution_time_master > execution_time and execution_time == 11:
        schedule_step = 0
        while not_finished(graph_trainer):
            schedulable_list = get_schedulable_tasks(graph_trainer, already_scheduled_trainer)
            preprocessed_matrix = numpy.column_stack([matrix_trainer, schedulable_list, already_scheduled_trainer])
            X_train = preprocessed_matrix
            Y_train = probabilist_solution_real(schedulable_list, student_value[schedule_step])
            model_master.fit(x=X_train, y=Y_train, batch_size=10, nb_epoch=1)
            real_solution = student_scheduling[schedule_step]
            update_dag(graph_trainer, real_solution)
            already_scheduled_trainer = already_scheduled_trainer + real_solution
            schedule_step += 1
    ##########################################################################################
    #                                     Testing Model                                      #
    ##########################################################################################

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

for i in range(nb_test):

    graph_test = create_broadcast(nb_storey)
    graph_test.solution = distance_pit(graph_test)
    graph = DAG(graph_size, 2)
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



# data.write(str(j*nb_epochs))
# data.write("\n")
# data.write(str(count_crash))
# data.write("  ")
# data.write(str(count_crash_test))
# data.write("\n")
# data.write(str(count_success_1))
# data.write("  ")
# data.write(str(count_success_1_test))
# data.write("\n")
# data.write(str(count_success_2))
# data.write("  ")
# data.write(str(count_success_2_test))
# data.write("\n")
# data.write(str(count_success_3))
# data.write("  ")
# data.write(str(count_success_3_test))
# data.write("\n")
# data.write(str(count_best))
# data.write("  ")
# data.write(str(count_best_test))
# data.write("\n")
data.write(str(total_ratio_before))
data.write("\n")
data.write(str(total_ratio))
data.write("\n")
print(total_ratio_before)
print(total_ratio)
data.write("\n")
data.write("\n")
