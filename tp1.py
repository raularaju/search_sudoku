from cgi import print_arguments
import sys
from typing import List
import queue
import copy
import time

list_algorithms: List[str] = ['B', 'I', 'U', 'A', 'G', 'D']

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
    for line in grid:
        print("".join(line), end=" ")

def put_num(grid, r_used, c_used, q_used, x, y, num):
    grid[x][y] = num
    r_used[x][int(num) - 1] = True
    c_used[y][int(num) - 1] = True
    q_used[(x//3) * 3 + (y//3)][int(num) - 1] = True

def is_ok_to_put_num(r_used, c_used, q_used, x , y, num):
    return not r_used[x][int(num) - 1] and not c_used[y][int(num) - 1] and not q_used[(x//3) * 3 + (y//3)][int(num) - 1] 

def expand_state(grid, r_used, c_used, q_used):
    children = []
    for i in range(0, 9):
        for j in range(0, 9):
                if (grid[i][j] == '0'):
                    for num in numbers:
                        if is_ok_to_put_num(r_used, c_used, q_used, i, j, num):
                            child_grid = copy.deepcopy(grid)
                            child_r_used = copy.deepcopy(r_used)
                            child_c_used = copy.deepcopy(c_used)
                            child_q_used = copy.deepcopy(q_used)
                            put_num(child_grid, child_r_used, child_c_used, child_q_used, i, j, num)
                            children.append((child_grid, child_r_used, child_c_used, child_q_used))
                    return children
    return children

        
def solve_bfs(grid, r_used, c_used, q_used, n_to_be_filled):
    q = queue.Queue()
    q.put((grid, r_used, c_used, q_used, n_to_be_filled))
    n_sts = 0
    while not q.empty():
        grid, r_used, c_used, q_used, n_to_be_filled = q.get()
        n_sts+=1
        if(n_to_be_filled == 0):
            break
        children = expand_state(grid,  r_used, c_used, q_used)
        for child_grid, child_r_used, child_c_used, child_q_used in children:
            q.put((child_grid, child_r_used, child_c_used, child_q_used, n_to_be_filled - 1))
    
    if n_to_be_filled > 0:
        return n_sts, None
    else:
        return n_sts, grid
        
    

def solve_dfs(grid, r_used, c_used, q_used, n_to_be_filled, max_depth):
    stack = []
    stack.append((grid, r_used, c_used, q_used, 0, n_to_be_filled))
    n_sts = 0
    while len(stack) > 0:
        grid, r_used, c_used, q_used, curr_depth, n_to_be_filled = stack.pop() 
        n_sts+=1
        if(n_to_be_filled == 0):
            break
        if(curr_depth >= max_depth):
            continue
        children = expand_state(grid,  r_used, c_used, q_used)
        for child_grid, child_r_used, child_c_used, child_q_used in children:
            stack.append((child_grid, child_r_used, child_c_used, child_q_used, curr_depth + 1, n_to_be_filled - 1))
    if n_to_be_filled > 0:
        return n_sts, None
    else:
        return n_sts, grid
        
def solve_ids(grid, r_used, c_used, q_used, n_to_be_filled):
    max_depth = 1
    n_sts, sol = solve_dfs(grid, r_used, c_used, q_used, n_to_be_filled, max_depth) 
    while sol == None or max_depth > 81:
        max_depth+=15
        n_sts, sol = solve_dfs(grid, r_used, c_used, q_used, n_to_be_filled, max_depth)
    return n_sts, sol

def solve_ucs(grid, r_used, c_used, q_used, n_to_be_filled):
    pq = queue.PriorityQueue()
    pq.put((0, grid, r_used, c_used, q_used, n_to_be_filled))
    n_sts = 0
    while not pq.empty():
        cost, grid, r_used, c_used, q_used, n_to_be_filled = pq.get()
        n_sts +=1
        if(n_to_be_filled == 0):
            break
        children = expand_state(grid,  r_used, c_used, q_used)
        for child_grid, child_r_used, child_c_used, child_q_used in children:
            pq.put((cost + 1, child_grid, child_r_used, child_c_used, child_q_used, n_to_be_filled - 1))
    if n_to_be_filled > 0:
        return n_sts, None
    else:
        return n_sts, grid
        
        
def heuristic1(grid, r_used, c_used, q_used, x, y):
    values = []
    if grid[x][y] != '0':
        return len(values), values
    for num in numbers:
        if is_ok_to_put_num(r_used, c_used, q_used, x, y, num):
            values.append(num)
    return len(values), values

def heuristic2(grid, r_used, c_used, q_used, x, y):
    num_constraints = 0
    for index in quadrants[x // 3][y // 3]:
        i, j = index
        if grid[i][j] == '0' and i != x and j != y:
            num_constraints += 1
    for i in range(9):
        if grid[x][i] == '0' and i != x:
            num_constraints += 1
    for i in range(9):
        if grid[i][y] == '0' and i != y:
            num_constraints += 1
    values = []
    for num in numbers:
        if is_ok_to_put_num(r_used, c_used, q_used, x, y, num):
            values.append(num)
    return 81 - num_constraints, values 

def expand_state2(grid, r_used, c_used, q_used, heuristic):
    children = []
    min_pos_values = 9 * 9 + 1
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
        children.append((min_pos_values, child_grid, child_r_used, child_c_used, child_q_used))
    return children

def solve_astar(grid,r_used, c_used, q_used, n_to_be_filled):
    pq = queue.PriorityQueue()
    initial_n_to_be_filled = n_to_be_filled
    pq.put((0, n_to_be_filled, grid, r_used, c_used, q_used))
    visited_states = set()
    n_sts = 0
    while not pq.empty():
        fn, n_to_be_filled, grid, r_used, c_used, q_used  = pq.get()
        n_sts+=1
        visited_states.add(str(grid))
        if(n_to_be_filled == 0):
            break
        children = expand_state2(grid, r_used, c_used, q_used, heuristic1)
        for min_pos_values, child_grid, child_r_used, child_c_used, child_q_used in children:
            if(str(child_grid) not in visited_states):
                pq.put((initial_n_to_be_filled - n_to_be_filled + 1 + min_pos_values, n_to_be_filled - 1, child_grid, child_r_used, child_c_used, child_q_used))  
    
    if n_to_be_filled > 0:
        return n_sts, None
    else:
        return n_sts, grid
        
    

def solve_gbfs(grid, r_used, c_used, q_used, n_to_be_filled):
    pq = queue.PriorityQueue() # Prioriy queue is not necessary because all children would have same priority
    pq.put((n_to_be_filled, grid, r_used, c_used, q_used))
    visited_states = set()
    n_sts = 0
    while not pq.empty():
        n_to_be_filled, grid, r_used, c_used, q_used  = pq.get() 
        n_sts+=1
        visited_states.add(str(grid))
        if(n_to_be_filled == 0):
            break
        children = expand_state2(grid, r_used, c_used, q_used, heuristic2)
        for _ , child_grid, child_r_used, child_c_used, child_q_used in children:
            if(str(child_grid) not in visited_states):
                pq.put((n_to_be_filled - 1 ,child_grid, child_r_used, child_c_used, child_q_used))
    
    if n_to_be_filled > 0:
        return n_sts, None
    else:
        return n_sts, grid
        
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
    
    start_time = time.perf_counter()
    if(algorithm == 'B'):
        n_sts, sol = solve_bfs(grid, r_used, c_used, q_used, n_to_be_filled)
    if(algorithm == 'I'):
        n_sts, sol = solve_ids(grid, r_used, c_used, q_used, n_to_be_filled)
    if(algorithm == 'U'):
        n_sts, sol = solve_ucs(grid, r_used, c_used, q_used, n_to_be_filled)
    if(algorithm == 'A'):
        n_sts, sol = solve_astar(grid, r_used, c_used, q_used, n_to_be_filled)
    if(algorithm == 'G'):
        n_sts, sol = solve_gbfs(grid, r_used, c_used, q_used, n_to_be_filled)
    if(algorithm == 'D'):
        n_sts, sol = solve_dfs(grid, r_used, c_used, q_used, n_to_be_filled, 81)
    end_time = time.perf_counter()
    exec_time = (end_time - start_time ) * 1000
    print(f"{n_sts} {exec_time:.0f}")
    print_grid(sol)

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