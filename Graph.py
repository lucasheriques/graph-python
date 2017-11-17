from collections import deque
import sys


class Graph:
    # reads graph from file
    def __init__(self):
        with open("input/in8.txt") as file:
            self.graph = [line.split() for line in file]
        for x in range(len(self.graph)):
            self.graph[x] = list(map(int, self.graph[x]))
        self.time = 0
        self.print_graph()

    # PARTE 1
    # prints number of vertices
    # complexity is O(1)
    def vertices_number(self):
        print("O grafo tem", len(self.graph), "vértice")
        return len(self.graph)

    # prints the number of edges in graph, divides by 2 because the graph is undirected
    # complexity is O(n^2)
    def edges_number(self):
        cont = 0
        for vertex in self.graph:
            for edge in vertex:
                if int(edge) > 0:
                    cont += 1
        print("O grafo tem", int(cont / 2), "arestas.")
        return cont

    # prints pending vertices of graph
    # complexity is O(n^2), where n is the number of vertices
    # O(n) for iterating each vertex times another O(n) for calculating the sum of edges
    def pending_vertices(self):
        cont = 0
        for vertex, edge in enumerate(self.graph):
            if sum(edge) == 1:
                # print("O vertice {} é pendente".format(vertex+1))
                cont += 1
        print("O grafo tem", cont, "vértices pendentes")
        return cont

    # prints null vertices of graph
    # complexity is O(n^2), where n is the number of vertices
    # O(n) for iterating each vertex times another O(n) for calculating the sum of edges
    def isolated_vertices(self):
        cont = 0
        for vertex, edge in enumerate(self.graph):
            if sum(edge) == 0:
                cont += 1
                # print("O vertice {} é isolado.".format(vertex + 1))
        print("O grafo tem", cont, "vértices isolados")
        return cont

    # prints degree of all vertices and neighbors
    # complexity is O(n^2)
    def vertices_degree_and_neighbors(self):
        for i, vertex in enumerate(self.graph):
            neighbors = []
            for e, edge in enumerate(vertex):
                # checks for auto loop
                if e != i and edge != 0:
                    neighbors.append(e+1)
            print("O vértice", i+1, "possui grau", len(neighbors), "e seus vizinhos são:", neighbors)

    # returns degree of vertex
    def vertex_degree(self, vertex):
        degree = 0
        for i in self.graph[vertex]:
            if i > 0:
                degree += 1
        return degree

    # prints number of components in graph
    # complexity is O(n^2)
    def components_number(self):
        visited = [False] * (len(self.graph))
        k = 0
        for v in range(len(visited)):
            if visited[v] is False:
                k += 1
                self.dfs_visit(v, visited, False)
        print("O grafo tem", k, "componente(s)")
        return k

    # prints if the graph is simple
    # complexity is O(n), where n is the number of vertices
    # adjacency matrix does not support parallel edges, thus this only check for auto loops
    def simple_graph(self):
        if all(edge[vertex] == 0 for vertex, edge in enumerate(self.graph)):
            print("O grafo é simples")
            return True
        else:
            print("O grafo não é simples")
            return False

    # prints if the graph is regular - all vertices have the same degree
    # complexity is O(n^2)
    def regular_graph(self):
        if all(self.vertex_degree(i) == self.vertex_degree(0) for i, vertex in enumerate(self.graph)):
            print("O grafo é regular")
            return True
        else:
            print("O grafo não é regular")
            return False

    # prints if the graph is null - there are no edges
    # complexity is O(n^2), because function all iterates in all vertices while checking if the condition is true
    # and sum sums edges of each vertex
    def null_graph(self):
        if all(sum(vertex) == 0 for vertex in self.graph):
            print("O grafo é nulo")
            return True
        else:
            print("O grafo não é nulo")
            return False

    # prints if the graph is complete - there's an edge between all vertices (Kn graphs)
    # complexity is O(n^2)
    def complete_graph(self):
        complete = True
        for vertex, edges in enumerate(self.graph):
            for i, edge in enumerate(edges):
                if i != vertex and edge == 0:
                    complete = False
        print("O grafo é completo") if complete else print("O grafo não é completo")
        return complete

    # prints if the graph is eulerian by checking if the degree of all vertices are even
    # complexity is O(n^2), because we need to loop all the n vertices
    # and then loop all the n lines of the adjacency matrix
    def eulerian_graph(self):
        eulerian = True
        for i, vertex in enumerate(self.graph):
            degree = 0
            for e, edge in enumerate(vertex):
                if edge != 0:
                    degree += 1
            if degree % 2 != 0:
                eulerian = False
        print("O grafo é euleriano") if eulerian else print("O grafo não é euleriano")
        return eulerian

    # prints if the graph is unicursal
    def unicursal_graph(self):
        odd_degrees = 0
        for i, vertex in enumerate(self.graph):
            degree = 0
            for e, edge in enumerate(vertex):
                if edge != 0:
                    degree += 1
            if degree % 2 != 0:
                odd_degrees += 1
        print("O grafo é unicursal") if odd_degrees == 2 else print("O grafo não é unicursal")
        return odd_degrees == 2

    # prints the complementary graph
    # complexity is O(n^2)
    def complementary_graph(self):
        complementary = []
        for vertex in self.graph:
            c_vertex = []
            for edge in vertex:
                c_vertex.append(0) if edge > 0 else c_vertex.append(1)
            complementary.append(c_vertex)
        self.print_graph(complementary)
        return complementary

    # checks if the graph is bipartite
    # TODO
    def bipartite_graph(self):
        bipartite = True
        colors = [-1] * len(self.graph)
        queue = deque()
        queue.append(0)
        colors[0] = 1

        while len(queue) > 0:
            v = queue.popleft()

            if self.graph[v][v] != 0:
                bipartite = False

            for e, weight in enumerate(self.graph[v]):
                if weight > 0 and colors[e] == -1:
                    colors[e] = 1 - colors[v]
                    queue.append(e)

                elif weight > 0 and colors[v] == colors[e]:
                    bipartite = False

        print("O grafo é bipartido") if bipartite else print("O grafo não é bipartido")
        return bipartite

    # PARTE 2
    # depth first search
    # complexity is O(n^2)
    def dfs(self):
        visited = [False]*(len(self.graph))
        for v in range(len(visited)):
            if visited[v] is False:
                self.dfs_visit(v, visited)
        print()

    def dfs_visit(self, v, visited, show=True):
        visited[v] = True
        print(v, end=' ') if show else None
        for i, edge in enumerate(self.graph[v]):
            if edge != 0 and visited[i] is False:
                self.dfs_visit(i, visited, show)

    # breadth first search
    # complexity is O(n^2) because we'll be looping each vertex, then each potential edge, which is also n
    def bfs(self, start=0):
        queue = deque()
        visited = [False]*(len(self.graph))

        for v in range(len(self.graph)):
            if visited[v] is False:
                visited[v] = True
                queue.append(v)
                for e, edge in enumerate(self.graph[v]):
                    if visited[e] is False and edge != 0:
                        visited[e] = True
                        queue.append(e)
            if queue:
                print(queue.popleft(), end =' ')
        print()

    # PARTE 3
    # check if graph is a tree
    # complexity is O(n^2)
    def is_cyclic(self, v, visited, parent):
        visited[v] = True

        for i, edge in enumerate(self.graph[v]):  # O(n)
            if edge != 0:
                if visited[i] is False:
                    if self.is_cyclic(i, visited, v) is True:  # O(n)
                        return True
                elif i != parent:
                    return True

        return False

    def is_tree(self):
        tree = True
        visited = [False] * len(self.graph)

        if self.is_cyclic(0, visited, -1) is True:
            tree = False

        if any(i is False for i in visited):
            tree = False

        print("O grafo é uma árvore") if tree else print("O grafo não é uma árvore")
        return tree

    # Prim
    # complexity is O(n^2), since we need to loop between each n vertex (O(n)), and then check
    # if there's an edge between each or vertex (O(n))
    def min_key(self, key, mstset):
        min_value = sys.maxsize
        min_index = -1

        for v in range(len(key)):
            if key[v] < min_value and v not in mstset:
                min_value = key[v]
                min_index = v

        return min_index

    def prim(self):
        print("Algoritmo de Prim - ordem de acesso dos vértices")
        mstset = []
        parent = [None] * len(self.graph)
        key = [sys.maxsize] * len(self.graph)
        key[0] = 0

        parent[0] = -1

        for x in range(len(self.graph)):  # O(n)
            v = self.min_key(key, mstset)   # O(n)
            mstset.append(v)

            for e, weight in enumerate(self.graph[v]):  # O(n)
                if 0 < weight < key[e] and e not in mstset:
                    parent[e] = v
                    key[e] = weight
            print(mstset)

    # Kruskal
    # complexity is O(n^3), since we have a while loop to guarantee we'll be getting
    # V - 1 edges (V is number of vertices), another O(n) for looping in each vertex
    # and another O(n) to chekc which edge exist and then get the lowest one
    def find(self, v, parent):
        if parent[v] == v:
            return v
        return self.find(parent[v], parent)

    def union(self, parent, rank, set1, set2):
        root1 = self.find(set1, parent)
        root2 = self.find(set2, parent)

        if rank[root1] < rank[root2]:
            parent[root1] = root2
        elif rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root2] = root1
            rank[root1] += 1

    def kruskal(self):
        graph = self.graph
        cont = 1
        parent = [x for x in range(len(graph))]
        rank = [0] * len(graph)
        while cont <= len(graph) - 1:  # O(n)
            min_value = sys.maxsize
            for i, edges in enumerate(graph):  # O(n)
                for v, weight in enumerate(edges):  # O(n)
                    if 0 < weight < min_value:
                        min_value = weight
                        v1 = i
                        v2 = v
            set1 = self.find(v1, parent)
            set2 = self.find(v2, parent)
            if set1 != set2:
                print("{} aresta ({}, {}): {}".format(cont, v1, v2, min_value))
                cont += 1
                self.union(parent, rank, set1, set2)
            graph[v1][v2] = graph[v2][v1] = -1

    # prints cut_vertices
    # complexity is O(n^2), since O(n) for each vertex, then another O(n) for checking
    # if vertex is a cut-vertex
    def cut_vetices_util(self, u, visited, ap, parent, low, disc):
        children = 0
        visited[u] = True
        disc[u] = self.time
        low[u] = self.time
        self.time += 1

        for v, weight in enumerate(self.graph[u]):  # O(n)
            if visited[v] is False and weight > 0:
                parent[v] = u
                children += 1
                self.cut_vetices_util(v, visited, ap, parent, low, disc)

                low[u] = min(low[u], low[v])

                # u is a root vertex with two or more children
                if parent[u] == -1 and children > 1:
                    ap.add(u)

                elif parent[u] != -1 and low[v] >= disc[u]:
                    ap.add(u)

            elif v != parent[u] and weight > 0:
                low[u] = min(low[u], disc[v])

    def cut_vertices(self):
        visited = [False] * len(self.graph)
        disc = [sys.maxsize] * len(self.graph)
        low = [sys.maxsize] * len(self.graph)
        parent = [-1] * len(self.graph)
        ap = set()

        for v, edges in enumerate(self.graph):  # O(n)
            if visited[v] is False:
                self.cut_vetices_util(v, visited, ap, parent, low, disc)

        print("O grafo possui", len(ap), "cut-vértice(s), sendo ele(s):", ap) if len(ap) > 0 else print("O grafo não possui cut-vértices.")
        return ap

    # complexity is O(n^2)
    def print_graph(self, graph=None):
        show = self.graph if graph is None else graph
        for row in show:
            for edge in row:
                print(edge, "\t", end='')
            print("")


g = Graph()
g.vertices_number()
g.edges_number()
g.pending_vertices()
g.isolated_vertices()
g.components_number()
g.vertices_degree_and_neighbors()
g.simple_graph()
g.null_graph()
g.regular_graph()
g.complete_graph()
g.eulerian_graph()
g.unicursal_graph()
g.complementary_graph()
g.bipartite_graph()
g.dfs()
g.bfs()
g.is_tree()
g.prim()
g.kruskal()
g.cut_vertices()
