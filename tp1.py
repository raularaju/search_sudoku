import sys
from typing import List
import queue
from tabulate import tabulate
import copy
list_algorithms: List[str] = ['B', 'I', 'U', 'A', 'G']

quadrants = [
    [[] for _ in range(3)] for _ in range(3)
    ]

for i in range(9):
    for j in range(9):
        quadrants[i // 3][j // 3].append((i,j))
        
numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

def usage(err_mes: str):
    print("Example: ")
    print("python tp1 [B | I | U | A | G] 107006450 025340008 060001070 053000029 610009800 000602007 001093200 008000000 040078591")
    raise Exception(err_mes)

def print_grid(grid):
    pass
    print(tabulate(grid, tablefmt="grid"))

def is_ok_to_put_num(grid, x , y, num):
    for i in range(9):
        if grid[x][i] == num and i != y:
            return False
        if grid[i][y] == num and i != x:
            return False
    for index in quadrants[x // 3][y // 3]:
         i, j = index
         if i != x and j != y and grid[i][j] == num:
             return False
    return True
def heuristic1(grid, i, j):
    pos_values = 100
    value = 1
    if grid[i][j] == '0':
        pos_values = 0
        for num in numbers:
            value = num
            if is_ok_to_put_num(grid, i,j, num):
                pos_values+=1
    return value, pos_values

def expand_state(state):
    grid = state[0]
    children = []
    for i in range(0, 9):
        for j in range(0, 9):
                if (grid[i][j] == '0'):
                    for num in numbers:
                        if is_ok_to_put_num(grid, i, j, num):
                            child_grid = copy.deepcopy(grid)
                            child_grid[i][j] = num
                            children.append(child_grid)
                    return children
    return children
def is_solution(st):
    pass
        
def solve_bfs(grid, n_to_be_filled):
    q = queue.Queue()
    q.put((grid, n_to_be_filled))
    n_sts = 0
    while not q.empty():
        st = q.get()
        n_sts+=1
        print(f"State {n_sts}")
        print_grid(st[0])
        if(st[-1] == 0):
            print("Solution found")
            return st[0]
        children = expand_state(st)
        for child in children:
            q.put((child, st[-1] - 1))
    if q.empty():
        raise Exception("BFS could not find any solution")

def solve_dfs(grid, n_to_be_filled, depth):
    stack = []
    stack.append((grid, 0, n_to_be_filled))
    n_sts = 0
    while len(stack) > 0:
        st = stack.pop() 
        n_sts+=1
        print(f"State {n_sts}")
        print_grid(st[0])
        if(st[-1] == 0):
            return st[0]
        if(st[1] >= depth):
            return None
        children = expand_state(st)
        for child in children:
            stack.append((child, st[1] + 1, st[-1] - 1))
        
def solve_ids(grid, n_to_be_filled):
    depth = 1
    sol = solve_dfs(grid, n_to_be_filled, depth) 
    while sol == None:
        print(f"Depth: {depth}")
        depth+=1
        sol = solve_dfs(grid, n_to_be_filled, depth)
    return sol

def expand_state2(state):
    grid = state[0]
    children = []
    min_pos_values = 10
    value = 1
    x = 0
    y = 0
    for i in range(9):
        for j in range(9):
            if grid[i][j] != '0':
                continue
            value, pos_values = heuristic1(grid, i,j) 
            if  pos_values < min_pos_values:
                min_pos_values = pos_values
                x = i
                y = j

    child_grid = copy.deepcopy(grid)
    child_grid[x][y] = value
    children = []
    for i in range(9):
        for j in range(9):
            if child_grid[i][j] != '0':
                continue
            children.append((heuristic1(child_grid, i,j)[-1], child_grid ))
    return children

def solve_ucs(grid, n_to_be_filled):
    pass

def solve_astar(grid, n_to_be_filled):
    pass

def solve_gbfs(grid, n_to_be_filled):
    q = queue.PriorityQueue()
    q.put((0, (grid, n_to_be_filled)))
    n_sts = 0
    while not q.empty():
        _, st = q.get()
        n_sts+=1
        print(f"State {n_sts}")
        print_grid(st[0])
        if(st[-1] == 0):
            print_grid(st[0])
            return st[0]
        children = expand_state2(st)
        for child in children:
            #print(child)
            q.put((child[0], (child[1], st[-1] - 1)))
    


def solve(algorithm : str, lines: List[str]):
    grid = []
    n_to_be_filled = 0
    for line in lines:
        n_to_be_filled += line.count('0')
        grid.append([char for char in line])
    #print_grid(grid)
    # res = is_ok_to_put_num(grid, 1, 0, '9')
    # print(res)
    if(algorithm == 'B'):
       solve_bfs(grid, n_to_be_filled)
    if(algorithm == 'I'):
        solve_ids(grid, n_to_be_filled)
    if(algorithm == 'U'):
        solve_ucs(grid, n_to_be_filled)
    if(algorithm == 'A'):
        solve_astar(grid, n_to_be_filled)
    if(algorithm == 'G'):
        solve_gbfs(grid, n_to_be_filled)

def main():
    if(len(sys.argv) - 1 != 10):
        usage("Número de argumentos errado")
    algorithm : str = sys.argv[1]
    
    if(algorithm not in list_algorithms):
        usage("Algoritmo não reconhecido")

    lines : List[str] = sys.argv[2:]
    solve(algorithm, lines)
    
if __name__ == "__main__":
    main()