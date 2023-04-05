from queue import PriorityQueue
from math import sqrt

class Problem:
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.goal_state = [[1,2,3],[4,5,6],[7,8,0]]
        self.operators = ['up', 'down', 'left', 'right']

    def get_initial_state(self):
        return self.initial_state

    def apply_operator(self, state, operator):
        new_state = [row[:] for row in state]
        for i, row in enumerate(new_state):
            if 0 in row:
                j = row.index(0)
                break
        try:
            if operator == 'up' and i > 0:
                new_state[i][j], new_state[i-1][j] = new_state[i-1][j], new_state[i][j]
            elif operator == 'down' and i < 2:
                new_state[i][j], new_state[i+1][j] = new_state[i+1][j], new_state[i][j]
            elif operator == 'left' and j > 0:
                new_state[i][j], new_state[i][j-1] = new_state[i][j-1], new_state[i][j]
            elif operator == 'right' and j < 2:
                new_state[i][j], new_state[i][j+1] = new_state[i][j+1], new_state[i][j]
        except:
            return None
        return new_state

    
# Node class: This class should contain the state, parent, operator, and
#     path_cost as instance variables. The operator variable should store 
#     the operator used to generate the current node, and the path_cost 
#     variable should store the cost of the path from the initial state 
#     to the current node.
class Node:
    def __init__(self, state, parent=None, operator=None, h=0):
        self.state = state
        self.parent = parent
        self.operator = operator
        self.children = []
        self.g = 0
        if parent:
            self.g = parent.g + 1
        self.h = h
        self.f = 0

    def add_child(self, child_node):
        self.children.append(child_node)

    def mapped_state(self):
        return tuple(map(tuple, self.state))

    def __repr__(self):
        return f"Node(f={self.f}, g={self.g}, h={self.h})"
    
    def __lt__(self, other):
        return self.f < other.f

# Tree class: This class should contain a root node as an instance variable.
class Tree:
    def __init__(self, problem):
        self.root = Node(problem.initial_state)
        self.problem = problem

    def get_root(self):
        return self.root

# initialize the default root node
puzzle = [[1, 2, 4], [4, 5, 0], [7, 8, 6]]
root = Problem(puzzle)

# initialize the frontier priority queue
frontier = PriorityQueue()
frontier.put((0, root))

# uniform_cost_search(problem): This function should implement Uniform Cost
#     Search on the given problem by using a priority queue to store the
#     nodes. The priority of a node should be the cost of the path from the
#     initial state to the node. This function should return the solution node.
def uniform_cost_search(problem):
    counter = 0
    tree = Tree(problem=problem)
    frontier = PriorityQueue()
    frontier.put(tree.get_root())  # (priority, node)
    explored = set()
    while not frontier.empty():
        curr_node = frontier.get()
        counter += 1
        if curr_node.state == problem.goal_state:
            break

        explored.add(curr_node.mapped_state())
        for operator in problem.operators:
            child_state = problem.apply_operator(curr_node.state, operator)
            if child_state is None:
                continue
            child_node = Node(child_state, parent=curr_node)
            if child_node.mapped_state() in explored:
                continue
            frontier.put(child_node)
            child_node.f = child_node.g
            curr_node.add_child(child_node)
                
    print_tree(curr_node)
    return counter

def frontier_contains_state(frontier, state):
    for _, node in frontier.queue:
        if node.state == state:
            return True
    return False

def get_incumbent(frontier, state):
    for _, node in frontier.queue:
        if node.state == state:
            return node
    return None

# misplaced_tile_heuristic(state, goal_state): This function should calculate
#     the Misplaced Tile heuristic for a given state and the goal state. The
#     Misplaced Tile heuristic is the number of tiles that are not in their 
#     correct position in the state.
def misplaced_tile_heuristic(state, goal_state):
    num_misplaced_tiles = 0
    for i in range(len(state)):
        for j in range(len(state[i])):
            if state[i][j] != goal_state[i][j]:
                num_misplaced_tiles += 1
    return num_misplaced_tiles

# euclidean_distance_heuristic(state, goal_state): This function should 
#     calculate the Euclidean Distance heuristic for a given state and the 
#     goal state. The Euclidean Distance heuristic is the sum of the 
#     Euclidean distances between each tile in the state and its correct 
#     position in the goal state.
def euclidean_distance(p, q):
    return sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)

def euclidean_distance_heuristic(state, goal_state):
    total_distance = 0
    for i in range(len(state)):
        for j in range(len(state[i])):
            tile = state[i][j]
            if tile != 0:
                goal_position = find_tile_position(goal_state, tile)
                distance = euclidean_distance((i, j), goal_position)
                total_distance += distance
    return total_distance
    
def find_tile_position(state, tile):
    for i in range(len(state)):
        for j in range(len(state[i])):
            if state[i][j] == tile:
                return (i, j)

# a_star_search(problem, heuristic): This function should implement 
#     A* Search with a given heuristic function on the given problem by 
#     using a priority queue to store the nodes. The priority of a node 
#     should be the sum of the cost of the path from the initial state 
#     to the node and the value of the heuristic function for the node's
#     state. This function should return the solution node.
def a_star_search(problem, heuristic):
    count = 0
    tree = Tree(problem=problem)
    start_node = tree.get_root()
    start_node.g = 0
    start_node.h = heuristic(start_node.state, problem.goal_state)
    start_node.f = start_node.g + start_node.h
    frontier = PriorityQueue()
    frontier.put(start_node)
    explored = set()
    while not frontier.empty():
        curr_node = frontier.get()
        count += 1
        if curr_node.state == problem.goal_state:
            break
        explored.add(curr_node.mapped_state())
        for operator in problem.operators:
            child_state = problem.apply_operator(curr_node.state, operator)
            if child_state is None:
                continue
            child_node = Node(child_state, parent=curr_node)
            child_node.h = heuristic(child_state, problem.goal_state)
            child_node.f = child_node.g + child_node.h
            curr_node.add_child(child_node)
            if child_node.mapped_state() not in explored:
                frontier.put(child_node)
    
    print_tree(curr_node)
    return count
def print_tree(node):
    if node.parent:
        print_tree(node.parent)
    print(node.state)

# Get the input from the user to choose the search algorithm 
# Call the appropriate search function with the Problem object and the 
#   chosen heuristic (if applicable) to get the solution node.
# Print the sequence of actions that led to the goal 
#   by following the parents of the solution node all the way up to 
#   the root of the tree.
# Print the number of nodes expanded, the maximum number of nodes in the
#   queue at any one time, and the depth of the goal node.
def main():
    
    u_cnt = 0
    m_cnt = 0
    e_cnt = 0

    print("Welcome to dboul001 8 puzzle solver.")
    print("Type '1' to use a default puzzle, or '2' to enter your own puzzle.")
    print("(Or if you are Daniel and you're testing, type \"t\")")
    
    puzzle_choice = input()
    if puzzle_choice == '1':
        puzzle = [[1, 0, 3], [4, 2, 6], [7, 5, 8]]
    elif puzzle_choice == '2':
        print("Enter your puzzle, use a zero to represent the blank")
        puzzle = []
        for i in range(3):
            row = input("Enter the {} row, use space or tabs between numbers: ".format(i + 1))
            puzzle.append([int(num) for num in row.split()])
    elif puzzle_choice == 't':
        print("Select which difficulty you want to test:")
        print("Trivial ----- 0")
        print("Very Easy --- 1")
        print("Easy -------- 2")
        print("Doable ------ 3")
        print("Oh Boy ------ 4")
        print("Impossible -- 5")

        test_case = input()
        if test_case == '0':
            puzzle = [[1, 2, 3],[4, 5, 6], [7, 8, 0]]
            root = Problem(puzzle)
            print("Uniform Cost Search Algo:")
            print("")
            u_cnt = uniform_cost_search(root)
            print("")

            print("A Star Search with Misplaced Tile Heuristic:")
            print("")
            m_cnt = a_star_search(root, misplaced_tile_heuristic)
            print("")

            print("A Star Search with Euclidean Distance Heuristic:")
            e_cnt = a_star_search(root, euclidean_distance_heuristic)
            print("")

            print("Uniform Nodes Expanded: ", u_cnt)
            print("Misplaced Nodes Expanded: ", m_cnt)
            print("Euclidean Nodes Expanded: ", e_cnt)
            exit(0)

        elif test_case == '1':
            puzzle = [[1, 2, 3],[4, 5, 6], [7, 0, 8]]
            root = Problem(puzzle)
            print("Uniform Cost Search Algo:")
            print("")
            u_cnt = uniform_cost_search(root)
            print("")

            print("A Star Search with Misplaced Tile Heuristic:")
            print("")
            m_cnt = a_star_search(root, misplaced_tile_heuristic)
            print("")

            print("A Star Search with Euclidean Distance Heuristic:")
            e_cnt = a_star_search(root, euclidean_distance_heuristic)
            print("")

            print("Uniform Nodes Expanded: ", u_cnt)
            print("Misplaced Nodes Expanded: ", m_cnt)
            print("Euclidean Nodes Expanded: ", e_cnt)
            exit(0)
            
        elif test_case == '2':
            puzzle = [[1, 2, 0],[4, 5, 3], [7, 8, 6]]
            root = Problem(puzzle)
            print("Uniform Cost Search Algo:")
            print("")
            u_cnt = uniform_cost_search(root)
            print("")

            print("A Star Search with Misplaced Tile Heuristic:")
            print("")
            m_cnt = a_star_search(root, misplaced_tile_heuristic)
            print("")

            print("A Star Search with Euclidean Distance Heuristic:")
            e_cnt = a_star_search(root, euclidean_distance_heuristic)
            print("")

            print("Uniform Nodes Expanded: ", u_cnt)
            print("Misplaced Nodes Expanded: ", m_cnt)
            print("Euclidean Nodes Expanded: ", e_cnt)
            exit(0)
            
        elif test_case == '3':
            puzzle = [[0, 1, 2],[4, 5, 3], [7, 8, 6]]
            root = Problem(puzzle)
            print("Uniform Cost Search Algo:")
            print("")
            u_cnt = uniform_cost_search(root)
            print("")

            print("A Star Search with Misplaced Tile Heuristic:")
            print("")
            m_cnt = a_star_search(root, misplaced_tile_heuristic)
            print("")

            print("A Star Search with Euclidean Distance Heuristic:")
            e_cnt = a_star_search(root, euclidean_distance_heuristic)
            print("")

            print("Uniform Nodes Expanded: ", u_cnt)
            print("Misplaced Nodes Expanded: ", m_cnt)
            print("Euclidean Nodes Expanded: ", e_cnt)
            exit(0)
            
        elif test_case == '4':
            puzzle = [[8, 7, 1],[6, 0, 2], [5, 4, 3]]
            root = Problem(puzzle)

            print("Uniform Cost Search Algo:")
            print("")
            u_cnt = uniform_cost_search(root)
            print("")

            print("A Star Search with Misplaced Tile Heuristic:")
            print("")
            m_cnt = a_star_search(root, misplaced_tile_heuristic)
            print("")

            print("A Star Search with Euclidean Distance Heuristic:")
            e_cnt = a_star_search(root, euclidean_distance_heuristic)
            print("")

            print("Uniform Nodes Expanded: ", u_cnt)
            print("Misplaced Nodes Expanded: ", m_cnt)
            print("Euclidean Nodes Expanded: ", e_cnt)
            exit(0)
            
        elif test_case == '5':
            puzzle = [[1, 2, 3],[4, 5, 6], [8, 7, 0]]
            root = Problem(puzzle)
            print("Uniform Cost Search Algorithm:")
            print("")
            u_cnt = uniform_cost_search(root)
            print("")

            print("A Star Search with Misplaced Tile Heuristic:")
            print("")
            m_cnt = a_star_search(root, misplaced_tile_heuristic)
            print("")

            print("A Star Search with Euclidean Distance Heuristic:")
            e_cnt = a_star_search(root, euclidean_distance_heuristic)
            print("")

            print("Uniform Nodes Expanded: ", u_cnt)
            print("Misplaced Nodes Expanded: ", m_cnt)
            print("Euclidean Nodes Expanded: ", e_cnt)
            exit(0)
            
        else:
            print("There is no test case for this. Please try again.")
            exit(0)

    print("Enter your choice of algorithm")
    print("Uniform Cost Search - enter '1'")
    print("A* with the Misplaced Tile heuristic. - enter '2'")
    print("A* with the Euclidean distance heuristic. - enter '3'")

    algorithm_choice = input()
    root = Problem(puzzle)
    if algorithm_choice == '1':
        algorithm = "uniform_cost"
    elif algorithm_choice == '2':
        algorithm = "misplaced"
    elif algorithm_choice == '3':
        algorithm = "euclidean"

    print("Solving puzzle using " + algorithm + " algorithm...")

    if algorithm == "uniform_cost":
        u_cnt = uniform_cost_search(root)
        print("Uniform Nodes Expanded: ", u_cnt)
    elif algorithm == "misplaced":
        m_cnt = a_star_search(root, misplaced_tile_heuristic)
        print("Misplaced Nodes Expanded: ", m_cnt)
    elif algorithm == "euclidean":
        e_cnt = a_star_search(root, euclidean_distance_heuristic)
        print("Euclidean Nodes Expanded: ", e_cnt)
    print("Goal!")

if __name__ == "__main__":
    main()

#Test Cases: 

# Trivial:      1 2 3 
#               4 5 6 
#               7 8 0

# Very Easy:    1 2 3 
#               4 5 6
#               7 0 8

# Easy:         1 2 0 
#               4 5 3 
#               7 8 6

# Doable:       0 1 2 
#               4 5 3 
#               7 8 6

# Oh Boy:       8 7 1 
#               6 0 2 
#               5 4 3

# Impossible:   1 2 3 
#               4 5 6 
#               8 7 0