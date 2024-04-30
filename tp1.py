import sys
from typing import List
import queue
from tabulate import tabulate
import copy
import time

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

def expand_state(grid):
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
        if(st[-1] == 0):
            print("Solution found")
            print(f"N states: {n_sts}")
            return st[0]
        children = expand_state(st[0])
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
        if(st[-1] == 0):
            print(f"N states: {n_sts}")
            return st[0]
        if(st[1] >= depth):
            return None
        children = expand_state(st[0])
        for child in children:
            stack.append((child, st[1] + 1, st[-1] - 1))
        
def solve_ids(grid, n_to_be_filled):
    depth = 1
    sol = solve_dfs(grid, n_to_be_filled, depth) 
    while sol == None:
        depth+=1
        sol = solve_dfs(grid, n_to_be_filled, depth)
    return sol

def solve_ucs(grid, n_to_be_filled):
    pq = queue.PriorityQueue()
    pq.put((0, grid, n_to_be_filled))
    n_sts = 0
    while not pq.empty():
        cost, grid, n_to_be_filled = pq.get()
        n_sts +=1
        if(n_to_be_filled == 0):
            print("Solution found")
            print_grid(grid)
            print(f"N states: {n_sts}")
            return grid
        children = expand_state(grid)
        for child in children:
            pq.put((cost + 1, child, n_to_be_filled-1))

        
def heuristic1(grid, x, y):
    pos_values = 100
    values = []
    if grid[x][y] == '0':
        pos_values = 0
        for num in numbers:
            if is_ok_to_put_num(grid, x, y, num):
                pos_values+=1
                values.append(num)
    return len(values), values

def heuristic2(grid, x, y):
    pos_values_c = 9
    pos_values_r = 9
    pos_values_q = 9
    values = []
    for num in numbers:
        is_ok_to_put = True
        for i in range(9):
             if grid[x][i] == num and i != y:
                pos_values_c -= 1
                is_ok_to_put = False
             if grid[i][y] == num and i != x:
                pos_values_r -= 1
                is_ok_to_put = False
        for index in quadrants[x // 3][y // 3]:
              i, j = index
              if i != x and j != y and grid[i][j] == num:
                    pos_values_q -= 1
                    is_ok_to_put = False
        if is_ok_to_put:
            values.append(num)
    return pos_values_c + pos_values_r + pos_values_c, values 

            
def expand_state2(grid, heuristic):
    children = []
    min_pos_values = 9 * 4
    min_values = []
    x = 0
    y = 0
    for i in range(9):
        for j in range(9):
            if grid[i][j] != '0':
                continue
            hn, values = heuristic(grid, i, j)
            if  hn < min_pos_values:
                min_values = values
                min_pos_values = len(values)
                x = i
                y = j

    children = []
    for value in min_values: 
        child_grid = copy.deepcopy(grid)
        child_grid[x][y] = value
        children.append(child_grid)
    return children

def solve_astar(grid, n_to_be_filled):
    pq = queue.PriorityQueue()
    pq.put((n_to_be_filled, grid ))
    visited_states = set()
    n_sts = 0
    while not pq.empty():
        n_to_be_filled, grid = pq.get() 
        n_sts+=1
        #print(f"State {n_sts}")
        #print_grid(grid)
        visited_states.add(str(grid))
        if(n_to_be_filled == 0):
            print(f"N states: {n_sts}")
            return grid
        children = expand_state2(grid, heuristic2)
        for child in children:
            if(str(child) not in visited_states):
                pq.put((n_to_be_filled - 1 ,child))
    print("Solution not found")


def solve_gbfs(grid, n_to_be_filled):
    pq = queue.PriorityQueue()
    pq.put((n_to_be_filled, grid ))
    visited_states = set()
    n_sts = 0
    while not pq.empty():
        n_to_be_filled, grid = pq.get() 
        n_sts+=1
        #print(f"State {n_sts}")
        #print_grid(grid)
        visited_states.add(str(grid))
        if(n_to_be_filled == 0):
            print(f"N states: {n_sts}")
            return grid
        children = expand_state2(grid, heuristic1)
        for child in children:
            if(str(child) not in visited_states):
                pq.put((n_to_be_filled -1 ,child))

def solve(algorithm : str, lines: List[str]):
    grid = []
    n_to_be_filled = 0
    for line in lines:
        n_to_be_filled += line.count('0')
        grid.append([char for char in line])
    sol = []
    if(algorithm == 'B'):
       sol = solve_bfs(grid, n_to_be_filled)
    if(algorithm == 'I'):
        sol = solve_ids(grid, n_to_be_filled)
    if(algorithm == 'U'):
        sol = solve_ucs(grid, n_to_be_filled)
    if(algorithm == 'A'):
        sol = solve_astar(grid, n_to_be_filled)
    if(algorithm == 'G'):
        sol = solve_gbfs(grid, n_to_be_filled)

def main():
    if(len(sys.argv) - 1 != 10):
        usage("Número de argumentos errado")
    algorithm : str = sys.argv[1]
    
    if(algorithm not in list_algorithms):
        usage("Algoritmo não reconhecido")

    lines : List[str] = sys.argv[2:]
    solve(algorithm, lines)
    
if __name__ == "__main__": 
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    exec_time = (end_time - start_time ) * 1000
    print(f"Exec. time: {exec_time:.0f}")