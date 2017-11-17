class Digraph:
    def __init__(self):
        with open("input.txt") as file:
            self.digraph = [line.split() for line in file]
        for x in range(len(self.digraph)):
            self.digraph[x] = list(map(int, self.digraph[x]))

    # prints if the digraph is balanced
    # complexity is O(n^2)
    def is_balanced(self):
        indegrees = [sum(x) for x in zip(*self.digraph)]
        outdegrees = [sum(x) for x in self.digraph]
        balanced = True if indegrees == outdegrees else False
        print("O dígrafo é balanceado") if balanced else print("O dígrafo não é balanceado")
        return balanced

    # check if digraph is regular
    # complexity is O(n^2)
    def is_regular(self):
        regular = False
        if self.is_balanced():
            indegrees = set(sum(x) for x in zip(*self.digraph))
            regular = True if len(indegrees) == 1 else False
        print("O digrafo é regular") if regular else print("O digrafo não é regular")

    # check if digraph is eulerian - all vertices have even degrees
    # complexity is
    def is_eulerian(self):
        regular = False
        if self.is_balanced():
            indegrees = set(sum(x) for x in zip(*self.digraph))
            regular = True if len(indegrees) == 1 else False
        print("O digrafo é euleriano") if regular else print("O digrafo não é euleriano")

    def transpose_graph(self):
        return [x for x in zip(*self.digraph)]

    def dfs(self, v, visited, graph):
        visited[v] = True

        for i, edge in enumerate(graph[v]):
            if visited[i] is False and edge != 0:
                self.dfs(i, visited, graph)

    # checks if digraph is strongly connected
    # complexity is O(n^2)
    def is_strongly_connected(self):
        sc = True
        visited = [False]*(len(self.digraph))
        self.dfs(0, visited, self.digraph)  # start on vertex 0
        if any(i is False for i in visited):
            sc = False

        comp = self.transpose_graph()
        visited = [False]*(len(self.digraph))
        self.dfs(0, visited, comp)
        if any(i is False for i in visited):
            sc = False

        print("O dígrafo é fortemente conexo") if sc else print("O dígrafo não é fortemente conexo")

    # complexity is O(n^2)
    def print_graph(self, graph=None):
        show = self.digraph if graph is None else graph
        for row in show:
            for edge in row:
                print(edge, "\t", end='')
            print("")

    # dijkstra algorithm: find the shortest paths to all vertices from the start vertex
    # complexity is O(n^2)
    def lowest_distance(self, dists, sptset):
        min_value = 999999

        for v in range(len(self.digraph)):
            if dists[v] <= min_value and v not in sptset:
                min_value = dists[v]
                min_index = v

        return min_index

    def dijkstra(self):
        dists = [999999] * len(self.digraph)
        dists[0] = 0
        sptset = set()

        while len(sptset) != len(self.digraph):
            u = self.lowest_distance(dists, sptset)
            sptset.add(u)
            for e, weight in enumerate(self.digraph[u]):
                print(u, ">", e, weight)
                if weight > 0 and e not in sptset and dists[e] > dists[u] + weight:
                    dists[e] = dists[u] + weight
                print(dists)

        for i, x in enumerate(dists):
            if x == 999999:
                dists[i] = -1

        print(dists)


d = Digraph()
d.dijkstra()
