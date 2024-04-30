import sys
from typing import List
import queue
#from tabulate import tabulate
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

""" def print_grid(grid):
    print(tabulate(grid, tablefmt="grid")) """

def put_num(grid, r_used, c_used, q_used, x, y, num):
    grid[x][y] = num
    r_used[x][int(num) - 1] = True
    c_used[y][int(num) - 1] = True
    q_used[(x//3) * 3 + (y//3)][int(num) - 1] = True

def is_ok_to_put_num(r_used, c_used, q_used, x , y, num):
    return not r_used[x][int(num) - 1] and not c_used[y][int(num) - 1] and not q_used[(x//3) * 3 + (y//3)][int(num) - 1] 

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
            print(f"N states: {n_sts}")
            return grid
        children = expand_state(grid)
        for child in children:
            pq.put((cost + 1, child, n_to_be_filled-1))

        
def heuristic1(grid, r_used, c_used, q_used, x, y):
    pos_values = 100
    values = []
    if grid[x][y] == '0':
        pos_values = 0
        for num in numbers:
            if is_ok_to_put_num(r_used, c_used, q_used, x, y, num):
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

            
def expand_state2(grid, r_used, c_used, q_used, heuristic):
    children = []
    min_pos_values = 9 * 4
    min_values = []
    x = 0
    y = 0
    for i in range(9):
        for j in range(9):
            if grid[i][j] != '0':
                continue
            hn, values = heuristic(grid, r_used, c_used, q_used, i, j)
            if  hn < min_pos_values:
                min_values = values
                min_pos_values = len(values)
                x = i
                y = j

    children = []
    for value in min_values: 
        child_grid = copy.deepcopy(grid)
        child_r_used = copy.deepcopy(r_used)
        child_c_used = copy.deepcopy(c_used)
        child_q_used = copy.deepcopy(q_used)
        put_num(child_grid, child_r_used, child_c_used, child_q_used, x, y, value)
        children.append((child_grid, child_r_used, child_c_used, child_q_used))
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


def solve_gbfs(grid, r_used, c_used, q_used, n_to_be_filled):
    pq = queue.PriorityQueue()
    pq.put((n_to_be_filled, grid, r_used, c_used, q_used))
    visited_states = set()
    n_sts = 0
    while not pq.empty():
        n_to_be_filled, grid, r_used, c_used, q_used  = pq.get() 
        n_sts+=1
        #print(f"State {n_sts}")
        #print_grid(grid)
        visited_states.add(str(grid))
        if(n_to_be_filled == 0):
            print(f"N states: {n_sts}")
            return grid
        children = expand_state2(grid, r_used, c_used, q_used, heuristic1)
        for child_grid, child_r_used, child_c_used, child_q_used in children:
            if(str(child_grid) not in visited_states):
                pq.put((n_to_be_filled -1 ,child_grid, child_r_used, child_c_used, child_q_used))

def solve(algorithm : str, lines: List[str]):
    grid = []
    n_to_be_filled = 0
    r_used = [[False for _ in range(9)] for _ in range(9)]
    c_used = [[False for _ in range(9)] for _ in range(9)]
    q_used = [[False for _ in range(9)] for _ in range(9)]
    for i in range(9):
        n_to_be_filled += lines[i].count('0')
        grid.append([char for char in lines[i]])
        for j in range(9):
            if lines[i][j] != '0':
                r_used[i][int(lines[i][j]) - 1] = True
                c_used[j][int(lines[i][j]) - 1] = True
                q_used[(i//3) * 3 + (j//3)][int(lines[i][j]) - 1] = True
    sol = []
    if(algorithm == 'B'):
       sol = solve_bfs(grid, r_used, c_used, q_used, n_to_be_filled)
    if(algorithm == 'I'):
        sol = solve_ids(grid, r_used, c_used, q_used, n_to_be_filled)
    if(algorithm == 'U'):
        sol = solve_ucs(grid, r_used, c_used, q_used, n_to_be_filled)
    if(algorithm == 'A'):
        sol = solve_astar(grid, r_used, c_used, q_used, n_to_be_filled)
    if(algorithm == 'G'):
        sol = solve_gbfs(grid, r_used, c_used, q_used, n_to_be_filled)
    print(sol)
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