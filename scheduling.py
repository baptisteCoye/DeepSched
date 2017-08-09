from DAG import *
from utils import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *


graph_size = 15
nb_epochs = 10000
nb_test = 0


##########################################################################################
#                            Deep Neural Network Definition                              #
##########################################################################################

model=Sequential()
model.add(Dense(512,activation='relu', input_dim=graph_size+2))
model.add(Dense(256,activation='linear'))
model.add(PReLU())
model.add(Dense(128, activation='linear'))
model.add(PReLU())
model.add(Dense(64, activation='relu'))
model.add(PReLU())
model.add(Dense(32, activation='linear'))
model.add(PReLU())
model.add(Dense(graph_size, activation='linear'))
model.add(Dense(1, activation='softplus'))
model.compile(RMSprop(lr=1e-4, decay=1e-8), loss="mean_squared_error")


##########################################################################################
#                                    Training Model                                      #
##########################################################################################


for i in range(nb_epochs):
    #############################################
    #               Initialisation              #
    #############################################
    graph = DAG(graph_size)
    graph.renumerotation()
    matrix = graph_to_matrix(graph)
    already_scheduled = numpy.zeros(graph_size)
    while not_finished(graph):

        #########################################
        #               Preprocessing           #
        #########################################
        solution = find_solution(graph)
        schedulable_list = get_schedulable_tasks(graph, already_scheduled)
        preprocessed_matrix = numpy.column_stack([matrix, schedulable_list, already_scheduled])
        print(preprocessed_matrix)
        print(schedulable_list)
        print(solution)
        training_matrix = transpose_matrix(preprocessed_matrix, graph_size, graph_size+2)
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
        update_matrix(matrix, solution, graph_size)
        update_dag(graph, solution)
        already_scheduled = already_scheduled + solution

##########################################################################################
#                                     Testing Model                                      #
##########################################################################################

for i in range(nb_test):
    graph = DAG(graph_size)
    graph.renumerotation()
    matrix = graph_to_matrix(graph)
    while not_finished(graph):
        solution = find_solution(graph)
        training_matrix = transpose_matrix(matrix, graph_size)
        X_train = training_matrix
        prediction = model.predict(X_train)
        print(matrix)
        print(prediction)
        update_matrix(matrix, solution, graph_size)
        update_dag(graph, solution)




