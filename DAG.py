import random
import numpy
from utils_dag import *
from utils import *

class DAG:

    def __init__(self, size, density):
        self.density = density
        self.size = size
        self.edges = self.graph_generation()
        self.solution = numpy.zeros(size)

    def graph_generation(self):
        edges = set()
        for i in range(self.density*self.size):
            edge = numpy.random.randint(0, self.size, 2)
            if edge[0] < edge[1]:
                edges.add((edge[0], edge[1]))
            elif edge[1] < edge[0]:
                edges.add((edge[1], edge[0]))
        for i in range(1, self.size-1):
            init = 0
            end = 0
            for j in range(1, self.size-1):
                if (j, i) in edges:
                    init += 1
                if (i, j) in edges:
                    end += 1
            if init == 0:
                edges.add((0, i))
            if end == 0:
                edges.add((i, self.size-1))
        return edges

    def renumerotation(self):
        vertex = list()
        for i in range(self.size):
            vertex.append(i)
        random.shuffle(vertex)
        new_edges = set()
        new_solution = numpy.zeros(self.size)

        for edge in self.edges:
            new_edges.add((vertex[edge[0]], vertex[edge[1]]))
        for i in range(self.size):
            new_solution[i] = self.solution[vertex[i]]
        self.edges = sorted(new_edges)
        self.solution = new_solution

    def lexicographic_order(self):
        lex_order = numpy.zeros(self.size)
        label = 1
        not_labeled = list()
        for i in range(self.size):
            not_labeled.append(i)
#labelling all the final nodes
        for i in range(self.size):
            connexions = 0
            for j in range(self.size):
                if (i, j) in self.edges:
                    connexions += 1
            if connexions == 0:
                lex_order[i] = label
                label += 1
                not_labeled.remove(i)
#labelling all the others:
        while not fully_labeled(lex_order, self.size):
            lexicographic_value_list = list()
            index_list = list()
            for i in not_labeled:
                if all_successor_labeled(lex_order, i, self.edges, self.size):
                    lexicographic_value_list.append(lexicographic_value(lex_order, i, self.edges, self.size))
                    index_list.append(i)
                    not_labeled.remove(i)
            sorted_list = sort_lexicographic(lexicographic_value_list, index_list)
            for i in range(len(sorted_list[1])):
                # print(sorted_list[0][i])
                lex_order[sorted_list[1][i]] = label
                label += 1
        return lex_order

    def scheduling_order(self,lex_order):
        scheduling = list()
        index = self.size
        while len(scheduling) < self.size:
            for j in range(self.size):
                if lex_order[j] == index:
                    scheduling.append(j)
                    index -= 1
        return scheduling


