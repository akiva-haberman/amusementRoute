# Algos Group Project
from copy import deepcopy
import itertools
import pathlib
import math
import time
import gc
import os

# Helpers ------------------------------------------------------------------------------------------------------------ #


# Reads input from terminal. First N is given (num_attractions) and then N lines follow
# Of 6 integers: x, y, o, c, u, t
def read_input():
    N = int(input())
    atts = [[int(n) for n in input().split()]
            for _ in range(N)]  # Get [x y o c u t] tuple for each attraction
    return N, atts


# Reads input from file, with exact same form
def read_file(path):
    if path.is_file():
        with open(path, "r") as curr_file:
            N = int(curr_file.readline().strip())
            atts = [[int(n) for n in curr_file.readline().strip().split()]
                    for _ in range(N)]
    return N, atts


# Whittle down list of attractions to ones which can actually be visited
def strip_attractions(N, attractions):
    new_N = N
    new_attractions = []
    for i, attraction in enumerate(attractions):
        x, y, o, c, u, t = attraction
        travel_t = distance(x, y, 200, 200)
        if travel_t > o:
            effective_o = travel_t
        else:
            effective_o = o

        if travel_t + effective_o + t > 1440 or travel_t > c:  # If for any nodes travel_t + o + t > 1440, disregard it
            new_N -= 1
        else:
            new_attractions.append(attraction)

    # This is all we really have to consider
    return new_N, new_attractions


# Euclidean distance (rounded to next highest integer)
def distance(x0, y0, x1, y1):
    return math.ceil(math.sqrt((x0 - x1)**2 + (y0 - y1)**2))


# (N+1)x(N+1) matrix, where dist_matrix[i][j] = dist_matrix[j][i] is the (symmetric) distance from i to j.
# "Node" N+1 is the start/end location, (200, 200)
# dist_matrix[i][i] = 0
def get_distances(graph):
    dist_mat = [[0 for _ in range(graph.v)] for _ in range(graph.v + 1)]

    for node in range(graph.v):
        xi, yi, _, _, _, _ = graph.meta_array[node]
        dist_mat[graph.v][node] = distance(xi, yi, 200, 200)
        dist_mat[node][node] = 0
        for j in range(node + 1, graph.v):
            xj, yj, _, _, _, _ = graph.meta_array[j]
            dist_mat[node][j] = dist_mat[j][node] = distance(xi, yi, xj, yj)

    for i in range(graph.v):
        dist_mat[i].append(dist_mat[graph.v][i])
    dist_mat[graph.v].append(0)

    return dist_mat


# The heuristic approach uses a greedy + brute force approach. These generate different sequences, this merges them
# Just for printing purposes, do NOT use this in further computation (self.t, self.loc etc. are not set)
def seq_append(winner1, winner2):
    ret_seq = Sequence()
    ret_seq.visited = winner1.visited + winner2.visited
    ret_seq.u = winner1.u + winner2.u
    return ret_seq


# Classes ------------------------------------------------------------------------------------------------------------ #
class Graph:

    def __init__(self, num_vertices):
        self.v = num_vertices
        # This array of metadata stores the attractions in the same order they're given to us
        self.meta_array = [0] * num_vertices
        # This stores the distance of each node from every other, including the center. See get_distances().
        self.adjacency_matrix = [[0 for _ in range(self.v + 1)]
                                 for _ in range(self.v + 1)]

    def add_node(self, node, attraction):
        self.meta_array[node] = attraction  # [x y o c u t]

    def pretty_print(self):
        for attract in self.meta_array:
            print(attract)


# Every sequence is a potential solution
class Sequence:

    def __init__(self):
        # This holds the nodes visited, in the order they were visited
        self.visited = []
        # Each sequence tracks its own utility as well as time
        self.u = self.t = 0
        # Every sequence starts at the center, this also gets updated as nodes are visited
        self.loc = (200, 200)

    def visit_attraction(self, node, brute):
        # Because the brute force algorithm works on a small subset of the original graph, it's necessary to tell
        # These sequence functions which graph they should be considering. It's easier to pass them a boolean (brute)
        # Rather than a whole graph object
        if brute:
            graph = brute_g
        else:
            graph = g

        # This checks whether a node can be safely added to the sequence. If so, it also updates self.t so we don't
        # Redo time/distance computations
        if not self.valid_attraction(node, brute):
            return False

        # Extract attraction info
        x, y, o, c, u, t = graph.meta_array[node]

        self.visited.append(node)
        self.u += u
        self.t += t
        self.loc = (x, y)
        return True

    def valid_attraction(self, node, brute):
        if brute:
            graph = brute_g
        else:
            graph = g

        x, y, o, c, u, t = graph.meta_array[node]

        # We sometimes want to start a sequence off at an arbitrary location without actually visiting a node. So, we
        # Always check the loc
        x1, y1 = self.loc

        # From current location to proposed node
        travel_t = distance(x, y, x1, y1)

        arrival = self.t + travel_t
        can_i_get_home = graph.adjacency_matrix[graph.v][node]

        if arrival < o:
            arrival = o  # Wait until opening time
        if arrival + t + can_i_get_home > 1440 or arrival > c or node in self.visited:  # Sequence is invalid
            return False

        self.t = arrival
        return True

    def pretty_print(self):
        print(len(self.visited))

        # Because we solve on a stripped down version of input graph, we have to translate back to the node labels
        # We were given. We also add 1, as input/ouput is 1-indexed but code is 0-indexed
        for visited_node in self.visited:
            new_attract = g.meta_array[visited_node]
            for j, old_attract in enumerate(all_attractions):
                if new_attract == old_attract:
                    print(j + 1, end=" ")
                    break
        print("")
        print("Utility:", self.u)
        print("Time:", self.t)


# MST Approach ------------------------------------------------------------------------------------------------------- #
# This returns a sequence based upon the tour through all nodes created from the MST.
# It does the TSP algorithm we learned in class but has one extra step for feasibility
def mstApproach():
    adjMat = g.adjacency_matrix
    mst = getMst(adjMat)  # generates the mst
    tour = mstToTour(mst,
                     g.v)  # turns the mst into a tour visiting every nodes
    # tour would returned to solve the traveling salesman problem,
    # but the aditional function below is needed to reduce the tour to its best feasible version.
    # This extra step which isn't needed for the TSP is why this approach isn't that good
    return tourToSequence(tour)


# input: an adjacency matrix G
# output: an array of edges in the MST, where an edge is a length 2 array
def getMst(G):
    solution = []
    N = len(G)
    selected_node = [0] * N

    no_edge = 0

    selected_node[0] = True

    # printing for edge and weight
    while (no_edge < N - 1):
        minimum = float("inf")
        a = 0
        b = 0
        for m in range(N):
            if selected_node[m]:
                for n in range(N):
                    if ((not selected_node[n]) and G[m][n] >= 0):
                        # not in selected and there is an edge
                        if minimum > G[m][n]:
                            minimum = G[m][n]
                            a = m
                            b = n
        solution.append([a, b])
        selected_node[b] = True
        no_edge += 1
    return solution


# Input: an mst (an array of edges) and the start node
# Output:  an array of the order through which you would go around the mst to
# visit every node. But the start node is omitted from that array.
def mstToTour(mst, start):
    tour = [start]

    # Basically if a node in the tour array is on an edge in the mst,
    # we add the other node on that edge to the tour array and remove the edge from the mst.
    # This is repeated until the mst is empty
    while (len(mst) > 0):
        for node in tour:
            for i in range(len(mst)):
                edge = mst[i]
                if (node == edge[0]):
                    tour.append(edge[1])
                    mst.pop(i)
                    break
                elif (node == edge[1]):
                    tour.append(edge[0])
                    mst.pop(i)
                    break
    tour.pop(0)  # getting rid of start node
    return tour


# Input: a tour through all the attractions
# Output: a Sequence object with as many nodes as possible, added in the order of the tour
def dumbTourToSequence(tour):
    seq = Sequence()
    for node in tour:
        seq.visit_attraction(node,
                             False)  # attempts to add the node to the sequence
    return seq


# Input: a tour through all the attractions
# The algorithm creates a tour using "dumbTourToSequence" and then tries
# removing a node and retrying the algorithm until it finds a locally
# optimal sequence, which is returned.
def tourToSequence(tour):
    seqOptimal = False
    curSeq = dumbTourToSequence(tour)
    while (not seqOptimal):
        seqOptimal = True
        for node in curSeq.visited:
            newTour = tour
            newTour.remove(node)
            newSeq = dumbTourToSequence(newTour)
            if (newSeq.u > curSeq.u):
                curSeq = newSeq
                seqOptimal = False
                break
    return curSeq


# Heuristic Approach ------------------------------------------------------------------------------------------------- #
def heuristic_approach(new_N):
    start_time = time.time()

    global brute_g
    winners = []

    global wrank_constant
    wrank_constant = 1
    global erank_constant
    erank_constant = 0

    while wrank_constant >= 0:

        current_N = new_N
        sequence = Sequence()
        while current_N > 10:
            sequence, valid_tags, current_N = heuristic_driver(sequence)

        valid_tags, brute_g_size = mark_valids(sequence)

        # Build brute force stuff
        brute_g = Graph(brute_g_size)

        new_attractions = []
        j = 0
        for i, attract in enumerate(g.meta_array):
            if valid_tags[i]:
                new_attractions.append(attract)
                brute_g.add_node(j, attract)
                j += 1

        brute_g.adjacency_matrix = get_distances(brute_g)

        full_seqs = brute_force(sequence.t, sequence.loc)
        full_seqs.sort(reverse=True, key=lambda seq: seq.u)

        winner1 = sequence
        winner2 = full_seqs[0]

        for i in range(len(winner2.visited)):
            visited_node = winner2.visited[i]
            new_attract = brute_g.meta_array[visited_node]
            for j, old_attract in enumerate(g.meta_array):
                if new_attract == old_attract:
                    winner2.visited[i] = j
                    break

        winners.append(seq_append(winner1, winner2))
        # winners.append(winner1)

        erank_constant += .1
        wrank_constant -= .1

    winners.sort(reverse=True, key=lambda winner: winner.u)
    return winners[0]


def heuristic_driver(seq):

    weights_i = assign_weight(seq)
    weights_i.sort(reverse=True,
                   key=lambda w: w[0])  # Sort in descending order of net_util

    earliest_end = []
    for i, attract in enumerate(g.meta_array):
        _, _, o, c, _, t = attract
        earliest_end.append([o + t, i])
    earliest_end.sort(key=lambda tup: tup[0])  # o + t

    ranks = []
    for i in range(g.v):
        for j, weight in enumerate(weights_i):  # Can be sped up
            if weight[1] != i:
                continue
            else:
                weight_rank = j
        for k, tup in enumerate(earliest_end):
            if tup[1] != i:
                continue
            else:
                end_rank = k

        rank_i = wrank_constant * weight_rank + erank_constant * end_rank
        ranks.append([rank_i, i])

    ranks.sort(key=lambda rank: rank[0])

    greedy_add_node(seq,
                    ranks)  # Tries to add nodes in descending order of rank
    valid_tags, num_valids = mark_valids(seq)

    return seq, valid_tags, num_valids


def assign_weight(seq):
    weights = [0 for _ in range(g.v)]
    curr_time = seq.t
    size = len(seq.visited)
    per_min = 104195 / 1440
    # print(per_min)
    if size == 0:
        curr_node = g.v  # If the sequence is empty, we're coming from "home"
    else:
        curr_node = seq.visited[size - 1]  # Otherwise, last node in sequence

    for to_j in range(g.v):
        _, _, o, _, u, t = g.meta_array[to_j]
        travel_to_j = g.adjacency_matrix[curr_node][to_j]
        arrival_at_j = curr_time + travel_to_j
        wait_time = o - arrival_at_j
        if wait_time < 0:
            wait_time = 0

        net_util = u - per_min * (travel_to_j + wait_time + t
                                  )  # Attempt to balance utility with cost
        weights[to_j] = [
            net_util, to_j
        ]  # Each node gets net_util stored  with index, so can identify after sort

    return weights


def mark_valids(seq):
    valid_tags = [False for _ in range(g.v)]
    num_valid = 0

    size = len(seq.visited)
    if size == 0:
        last = g.v
    else:
        last = seq.visited[size - 1]

    for node in range(g.v):
        if node in seq.visited:
            continue

        _, _, o, _, _, t = g.meta_array[node]
        travel_t = g.adjacency_matrix[last][node]
        arrival = seq.t + travel_t
        if arrival < o:
            arrival = o
        time_to_home = g.adjacency_matrix[g.v][node]

        if arrival + t + time_to_home <= 1440:
            valid_tags[node] = True
            num_valid += 1

    return valid_tags, num_valid


def greedy_add_node(s, ranks):
    rank_index = 0
    while not s.visit_attraction(ranks[rank_index][1], False):
        if rank_index == g.v - 1:  # No nodes were valid
            break
        rank_index += 1


# Brute Forcer ------------------------------------------------------------------------------------------------------- #
def brute_force(start_t, start_loc):
    perms = itertools.permutations(range(brute_g.v), brute_g.v)
    finished_seqs = []
    for perm in perms:
        perm_seq = seq_builder(perm, start_t, start_loc)
        finished_seqs.append(perm_seq)

    return finished_seqs


def seq_builder(perm, start_t, start_loc):
    perm_sequence = Sequence()
    perm_sequence.t = start_t
    perm_sequence.loc = start_loc
    for node in perm:
        if not perm_sequence.visit_attraction(node, True):
            break
    return perm_sequence


# Dynamic Brute Force Approach --------------------------------------------------------------------------------------- #

# This is an upgraded version of:
# This algorithm starts with the 0-sequence (aka visiting no attractions) and
# then iteratively creates all k+1 length sequences by adding 1 node to the
# k length sequences in every possible way. So it tries every possible sequence.
# The best of these sequences is returned.

# The simple idea behind the upgrade is this:
# Suppose that [a, b, c] and [b, a, c] are two paths where a, b, c are attractions
# and that [b, a, c] takes more time. Then it is necessarily worse than [a, b, c]
# and can be forgotten about going forward. That is because these paths go through
# the same nodes and end at the same spot, but one basically just has a head start.
# So, everytime this algorithm discovers a new path, it only stores it if it is
# faster than current fastest path which goes through the same nodes and ends up
# at the same spot.


def dynamicDictPlusApproach():
    N = g.v

    # use MST approach to come up with approximate bestSequence,
    # which will help eliminate suboptimal solutions earlier.
    # The heuristic_approach will (theoretically) help eliminate even more suboptimal solutions
    # even earlier, but takes much longer
    if N <= 100:
        bestSequence = mstApproach()
    else:
        bestSequence = heuristic_approach(N)

    # The kth element of the paths array is a dictionary storing all the k length paths.
    # The dictionary has values of the k length sequences, making this approach
    # very similar to the dynamicArrApproach. However, there is now also a key which,
    # for any k length path, is the ordered pair of:
    # first: set of all the nodes visited thus far except for the last one
    # second: the last node it visits
    paths = [{
        (frozenset({}), None): Sequence()
    }]  # the kth element is the array of paths of length k

    # iterating through k, the length of the paths currentely being made
    for k in range(1, N + 1):
        kLengthPaths = {}
        prevPaths = paths[k - 1]  # the dictionary of paths of length k-1

        # iterate through each k-1 length path
        for pathKey in prevPaths:
            prevPath = prevPaths.get(pathKey)
            visited = prevPath.visited
            utility = prevPath.u
            path_t = prevPath.t
            loc = prevPath.loc

            # if the maximum utility achievable for the prevPath is less than
            # the utility of the current best sequence
            if utility + maxUtil(k, path_t) <= bestSequence.u:
                # Then forget about this path
                continue

            pathDead = True
            # Now iterating through each node
            for newNode in range(N):
                # if the node hasn't already been visited by the prevPath
                if newNode not in visited:
                    # then make a duplicate sequence and try adding the new node
                    newPath = Sequence()
                    newPath.visited = list(visited)
                    newPath.u = utility
                    newPath.t = path_t
                    newPath.loc = loc
                    # if we succeed in adding the new node (False just means it's not on the brute force graph)
                    if newPath.visit_attraction(newNode, False):
                        # then see if the path is better than its permutations
                        # which end in the same spot.
                        key = (frozenset(visited), newNode)
                        # if there is not a better permutation of newPath in the dictionary
                        if not kLengthPaths.get(
                                key) or newPath.t < kLengthPaths.get(key).t:
                            kLengthPaths[key] = newPath
                            pathDead = False
            # if path can't continue further
            if pathDead:
                # see if it is better than the current best sequnce
                if utility > bestSequence.u:
                    # if so update bestSequence
                    bestSequence = prevPath
        paths.append(kLengthPaths)  # store the dictionary of k length paths

    # see if any of the N-1 length paths are better than the bestSequence
    nlengthPaths = paths[-1]
    for path in nlengthPaths:
        seq = nlengthPaths.get(path)
        if seq.u > bestSequence.u:
            bestSequence = seq

    return bestSequence


# returns an upperbound on the utility that can be acheived by a path which has
# gone through "numVisited" nodes and used "time" time
def maxUtil(numVisited, time):
    nodesLeft = g.v - numVisited - 1  # the max num of additional nodes which can be visited without regard to time
    timeLeft = 1440 - time
    # find largest index of bestCaseArr which obeys time and nodesLeft constraint
    # the max additional utility acheivable according ot this array is returned
    for i in range(1, nodesLeft):
        if timeLeft < bestCaseArr[i][0]:
            return bestCaseArr[i - 1][1]
    return bestCaseArr[nodesLeft][1]


# This function loads the global variable bestCaseArr.
# The kth element of bestCaseArr is a length 2 arrays containing a
# min time and max utility of a k length path, which can be used to eliminate
# suboptimal paths early
def buildBestCaseArray():
    global bestCaseArr
    bestCaseArr = [[0, 0]]
    nodeUtils = []
    nodeTimes = []
    for i in range(len(g.meta_array)):
        nodeUtils.append([i, g.meta_array[i][4]])
        nodeTimes.append([i, g.meta_array[i][5]])
    nodeUtils.sort(reverse=True, key=lambda el: el[1])
    nodeTimes.sort(key=lambda el: el[1])
    mst = getMst(g.adjacency_matrix)
    maxUtil = 0
    minTime = 0
    for k in range(1, g.v):
        maxUtil += nodeUtils[k - 1][1]
        minInd = mst.index(
            min(mst, key=lambda edge: g.adjacency_matrix[edge[0]][edge[1]]))
        i, j = mst[minInd]
        mst.pop(minInd)
        minTime += g.adjacency_matrix[i][j] + nodeTimes[k - 1][1]
        bestCaseArr.append([minTime, maxUtil])
    return bestCaseArr


#  ------------------------------------------------------------------------------------------------------- #


# Read input, get all of the important global vars up and running, etc.
def loadEverything(file_path=None):
    global all_attractions
    if file_path is None:
        N, all_attractions = read_input()
    else:
        N, all_attractions = read_file(file_path)

    new_N, attractions = strip_attractions(
        N, all_attractions)  # To help with larger N's

    # Build graph as global graph
    global g
    g = Graph(new_N)
    for i, attraction in enumerate(attractions):
        g.add_node(i, attraction)

    # Build dist_matrix as global
    dist_matrix = get_distances(g)
    g.adjacency_matrix = dist_matrix
    buildBestCaseArray()


# Because we solve on a stripped down version of input graph, we have to translate back to the node labels
# We were given. We also add 1, as input/ouput is 1-indexed but code is 0-indexed
def translate_nodes(seq):
    for i in range(len(seq.visited)):
        visited_node = seq.visited[i]
        new_attract = g.meta_array[visited_node]
        for j, old_attract in enumerate(all_attractions):
            if new_attract == old_attract:
                seq.visited[i] = j + 1
                break


def testApproaches(file_path=None):
    new_N = g.v
    print("--------------------- Dynamic Dict Plus ---------------------")
    start = time.time()
    # winner = dynamicDictPlusApproach()
    winner = heuristic_approach(g.v)
    winner.pretty_print()
    print("Utility:", winner.u)
    print("Time: ", winner.t + g.adjacency_matrix[new_N][winner.visited[-1]])
    print("Runtime:", time.time() - start)
    print("")
    if file_path is None:
        return
    # For each input file (ex. BFS_small1.in) creates a corresponding output (BFS_small1.out) in a new directory
    out_file_path = file_path.parent.parent / 'output_files' / (
        file_path.stem + '.out')
    translate_nodes(winner)
    output = [str(len(winner.visited)) + "\n", str(winner.visited)]
    with open(out_file_path, "w+") as fp:
        fp.writelines(output)


if __name__ == '__main__':
    print("")
    start_time = time.time()
    yes = ["y", "Y", "yes", "yeah", "yup"]
    arg = input("Directory? (Y/N) > ")
    if arg not in yes:
        loadEverything()
        testApproaches()
    else:
        direct = input("Directory Name > ")
        input_dir = pathlib.Path(direct)
        os.mkdir(input_dir.parent / 'output_files')
        for path in input_dir.iterdir():
            if path.is_file():
                loadEverything(path)
                testApproaches(path)

    print("--- %s seconds ---" % (time.time() - start_time))
    gc.collect()
    print("")