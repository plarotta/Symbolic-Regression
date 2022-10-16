import numpy as np
import random
import matplotlib.pyplot as plt 
import math
import time


def add(x,y):
	return(x+y)

def subtract(x,y):
	return(x-y)

def multiply(x,y):
	return(x*y)

def divide(x,y):
	return(x/y)

def sin(x):
	return(np.sin(x))

def cos(x):
	return(np.cos(x))


def tree_solver(tree, root_idx=1):
	right_done = None
	left_done = None
	if tree[root_idx] == sin or tree[root_idx] == cos:
		try:
			left_type = type(tree[root_idx*2])
			left_done = left_type == float or left_type == str
			if not left_done:
				return(tree_solver(tree, root_idx*2))
		except IndexError:
			try:
				right_type = type(tree[root_idx*2+1])
				right_done = right_type == float or right_type == str
				if not right_done:
					return(tree_solver(tree, root_idx*2))
			except IndexError:
				return(False)
	else:
		try:
			left_type = type(tree[root_idx*2])
			right_type = type(tree[root_idx*2+1])
			left_done = left_type == float or left_type == str
			right_done = right_type == float or right_type == str
		except IndexError:
			return(False)		

	if left_done and right_done:
		return(True)
	elif right_done:
		if tree[root_idx] == sin or tree[root_idx] == cos:
			return(True)
		else:
			return(True and tree_solver(tree, root_idx*2))
	elif left_done:
		if tree[root_idx] == sin or tree[root_idx] == cos:
			return(True)
		else:
			return(True and tree_solver(tree, root_idx*2+1))
	
	else:
		return(tree_solver(tree, root_idx*2) and tree_solver(tree, root_idx*2+1))



# tree = [None, sin, divide, None, subtract, 2, None, None, 'x', -10]
def evaluator(tree, val):
    wtree = tree.copy()
    for idx in reversed(range(1,len(wtree))):
        if wtree[idx] == 'x':
            wtree[idx] = val
        elif wtree[idx] == sin or wtree[idx] == cos:
            wtree[idx] = wtree[idx](wtree[idx*2])
        elif callable(wtree[idx]):
            wtree[idx] = wtree[idx](wtree[idx*2], wtree[idx*2+1])
        else:
            pass
    return(wtree[1])

def read_data():
  f = open("data2022_Bronze.txt", "r")
  data = []
  while(True):
      line = f.readline()
      if not line:
          break
      line = [float(x) for x in line.strip().split(",")]
      data.append(tuple(line))
  f.close
  return(data)

def get_tree_depth(tree):
    size = sum([i != None for i in tree])
    return(math.floor(math.log(size,2))+1)

def random_search(iterations):
    dat = read_data()
    x_vals = [tup[0] for tup in dat]
    y_vals = [tup[1] for tup in dat]
    
    best_func = [None, add, 'x', 1]
    best_fitness = 100000
    best_fitnesses = []
    best_x = []
    best_y = []
    n = 0
    while n < iterations:
        tree = fgen(4)
        y_pred = []
        x_pred = []
        x_indices = []
        for i in range(len(x_vals)):
            try:
                y_pred.append(evaluator(tree,x_vals[i]))
                x_pred.append(x_vals[i])
                x_indices.append(i)
            except ZeroDivisionError:
                continue
        if len(y_pred)==0:
            continue
        fitness = np.mean([(y_vals[x_indices[i]]-y_pred[i])**2 for i in range(len(x_indices))])/len(x_indices)
        if fitness < best_fitness:
            best_func = tree
            best_fitnesses.append(fitness)
            best_fitness = fitness
            
            best_y = y_pred
            best_x = x_pred
        n+=1
    
    print(best_fitness)
    print(best_func)
    print(best_fitnesses)

    plt.plot(x_vals, y_vals)
    plt.plot(best_x, best_y)
    plt.show()  

    return(best_func)

def make_child(end=False):
    if end:
        choices = ["x","num"]
        child = random.choices(choices, [1,1])[0]
        child = random.uniform(-10,10) if child == "num" else child
    else:
        ops = ["+", "-", "*", "/", "sin", "cos"]
        ops_and_vals = ops + ['num'] + ['x']
        weights = [1,1,1,1,1,1] + [1] + [1]
        op_functions = {"+": add, "-": subtract, "*": multiply, "/": divide, "sin": sin, "cos": cos}
        child = random.choices(ops_and_vals, [i/1000 for i in weights], k=1)[0]
        if child in ops:
            child = op_functions[child]
        elif child == "num":
            child = random.uniform(-10,10)
        else:
            pass

    return(child)



def fgen(max_depth):
    done = False
    n = 0
    while not done:
        n+=1
        last_gen = 2**(max_depth-1)
        tree = [None for i in range(2**(max_depth+1))]
        tree[1] = add
        for idx in range(1, len(tree)):
            node = tree[idx]
            try:
                if callable(node):
                    if idx >= last_gen:
                        if node == sin or node == cos:
                            tree[idx*2] = make_child(True)
                        else:
                            tree[idx*2] = make_child(True)
                            tree[idx*2+1] = make_child(True)
                    else:
                        if node == sin or node == cos:
                            tree[idx*2] = make_child()
                        else:
                            tree[idx*2] = make_child()
                            tree[idx*2+1] = make_child()
            except IndexError:
                    break
        if "x" in tree:
            done = True
    end_idx = next(x for x in reversed(range(len(tree))) 
                          if tree[x] is not None)
    tree = tree[:end_idx+1]
    return(tree)

def get_fitness(indiv, dataset):
    x_vals = [tup[0] for tup in dataset]
    y_vals = [tup[1] for tup in dataset]
    y_pred = []
    x_pred = []
    x_indices = []
    for i in range(len(x_vals)):
        try:
            y_pred.append(evaluator(indiv,x_vals[i]))
            x_pred.append(x_vals[i])
            x_indices.append(i)
        except ZeroDivisionError:
            continue
    y_vals = [y_vals[i] for i in x_indices]
    return(mean_squared_error(y_vals, y_pred)/1000)




def GP(n_gen, init_population):
    #init population [done]
    #choose parent pairs [done]
    #cross all parent pairs [done]
    #mutate cross results (offspring)
    #deterministic crowding competition
    #fitness evaluations
    mut_rate = 0.1
    initial_population = [fgen(4) for i in range(init_population)]
    fitnesses = [1 for i in initial_population]
    pool = []
    pool_size = 0
    for i in range(n_gen):
        while pool_size < len(initial_population):
            couple = np.random.choice(initial_population, size=2, p=fitnesses, replace=False)
            children = cross_parents(couple, mut_rate)
            replacements = det_crowding(couple, children)
            [pool.append(i) for i in replacements]
        initial_population = pool
        fitnesses = [get_fitness(individual, dataset) for individual in initial_population]
        pool = 0
        pool_size = 0 

    final_population = [x for y, x in sorted(zip(fitnesses, initial_population))]

    return(final_population)
    


def mutator(tree, mut_rate):
    mutation = random.choices([True, False], weights=[mut_rate, 1], k=1)[0]
    if not mutation:
        return(tree)
    
    for idx, val in enumerate(tree):
        if 





def get_subnodes(tree, idx):
    #returns indices of subtrees, excluding the idx node index
    if type(tree[idx]) in (str, float):
        return([])
    unitary = True if tree[idx] in [sin,cos] else False
    if unitary:
        left_child_good = type(tree[idx*2]) in (str, float)
        if left_child_good:
            return([idx*2])
        else:
            return([idx*2] + get_subnodes(tree,idx*2))
    else:
        try:
            left_child_good = type(tree[idx*2]) in (str, float)
            right_child_good = type(tree[idx*2+1]) in (str, float)
            if left_child_good and right_child_good:
                return([idx*2, idx*2+1])
            elif left_child_good:
                return([idx*2] + [idx*2+1] + get_subnodes(tree,idx*2+1))
            elif right_child_good:
                return([idx*2+1] + [idx*2] + get_subnodes(tree,idx*2))
            else:
                return([idx*2] + [idx*2+1] + get_subnodes(tree,idx*2) + get_subnodes(tree,idx*2+1))
        
        except IndexError:
            return([])




def cross_parents(couple):
    bad = True
    while bad:
        left_idx = random.choice(range(2, len(couple[0])))
        right_idx = random.choice(range(2, len(couple[1])))
        #print("indices: , ", left_idx, right_idx)

        
        if (couple[0][left_idx] == None) or (couple[1][right_idx] == None):
            bad = True
        else:
            bad = False

    print("indices: , ", left_idx, right_idx)
    l_idx_depth = math.floor(math.log(left_idx)/math.log(2))
    r_idx_depth = math.floor(math.log(right_idx)/math.log(2))
    delta = right_idx-left_idx
    left_nodes = [left_idx] + get_subnodes(couple[0], left_idx)
    right_nodes = [right_idx] + get_subnodes(couple[1], right_idx)

    left_depth = get_tree_depth(couple[0])
    right_depth = get_tree_depth(couple[1])

    l_nodes_trans = []
    r_nodes_trans = []
    new_r_tree = couple[1].copy()
    new_l_tree = couple[0].copy()


    for i in left_nodes:
        new_l_tree[i] = None
        i_depth = math.floor(math.log(i)/math.log(2))
        l_nodes_trans.append(i + delta* 2 **(i_depth -l_idx_depth))
    
    for j in right_nodes:
        new_r_tree[j] = None
        j_depth = math.floor(math.log(j)/math.log(2))
        #print(left_nodes)
        #print(j_depth, r_idx_depth)
        r_nodes_trans.append(j - delta* 2 **(j_depth -r_idx_depth))


    for i in range(len(l_nodes_trans)):
        try:
            new_r_tree[l_nodes_trans[i]] = couple[0][left_nodes[i]]
        except IndexError:
            for j in range(l_nodes_trans[i]+1 - len(new_r_tree)):
                new_r_tree.append(None) 
            new_r_tree[l_nodes_trans[i]] = couple[0][left_nodes[i]]
            

    for i in range(len(r_nodes_trans)):
        try:
            new_l_tree[r_nodes_trans[i]] = couple[1][right_nodes[i]]
        except IndexError:
            for j in range(r_nodes_trans[i]+1 - len(new_l_tree)):
                new_l_tree.append(None) 
            new_l_tree[r_nodes_trans[i]] = couple[1][right_nodes[i]]


    return(new_l_tree, new_r_tree)
    








if __name__ == "__main__":
    

    #print(tree_editor([0,sin], [2]))
    #random_search(1000)
    #print(function_generator())
    #a = fgen(6)
    #print(a)
    #print(tree_solver(a[0]))
    tree1 = fgen(5)
    tree2 = fgen(5)
    print("tree1 valid: ", tree_solver(tree1, 1))
    print("tree2 valid: ", tree_solver(tree2, 1))
    print([tree_solver(i) for i in cross_parents([tree1,tree2])])
    