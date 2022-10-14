import numpy as np
import random
import cython
import matplotlib.pyplot as plt 
import math


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

def function_generator():
	operator_list = ["+", "-", "*", "/", "sin", "cos"]
	op_functions = {"+": add, "-": subtract, "*": multiply, "/": divide, "sin": sin, "cos": cos}
	tree = [None]
	tree.append(op_functions[random.choice(operator_list)])
	tree_not_finished = True
	while tree_not_finished:
		edits = []
		for index in range(1, len(tree)):
			if callable(tree[index]):
				if tree[index] == sin or tree[index] == cos:
					try:
						child_l = tree[2*index]
						if child_l == None:
							edits.append(2*index)
							continue
						else:
							continue
					except IndexError:
						edits.append(2*index)
						continue

					try:
						child_r = tree[2*index+1]
						if child_r == None:
							edits.append(2*index+1) 
							continue
						else:
							continue
					except IndexError:
						edits.append(2*index+1) 
						continue
				else:	

					try:
						child_r = tree[2*index+1]
						if child_r == None:
							edits.append(2*index+1) 
					except IndexError:
						edits.append(2*index+1) 


					try:
						child_l = tree[2*index]
						if child_l == None:
							edits.append(2*index)
					except IndexError:
						edits.append(2*index)

		tree = tree_editor(tree, edits); print(len(tree))
		tree_not_finished = not tree_solver(tree, 1)
	return(tree)




def tree_editor(tree, edits):
    ops = ["+", "-", "*", "/", "sin", "cos"]
    ops_and_vals = ops + ['num'] + ['x']
    weights = [1,1,1,1,1,1] + [1] + [1]
    op_functions = {"+": add, "-": subtract, "*": multiply, "/": divide, "sin": sin, "cos": cos}
    working_tree = tree.copy()
    for idx in edits:
        try:
            if working_tree[idx] == None:
                elem = random.choices(ops_and_vals, [i for i in weights], k=1)[0]
                if elem in ops:
                    working_tree[idx] = op_functions[elem]
                elif elem == "num":
                    working_tree[idx] = random.uniform(-10,10)
                else:
                    working_tree[idx] = elem
        except IndexError:
            for i in range(idx+1 - len(working_tree)):
                working_tree.append(None) 
            elem = random.choices(ops_and_vals, [i for i in weights], k=1)[0]
            if elem in ops:
                working_tree[idx] = op_functions[elem]
            elif elem == "num":
                working_tree[idx] = random.uniform(-10,10)
            else:
                working_tree[idx] = elem

    return(working_tree)



def tree_solver(tree, root_idx):
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
            # try:
            wtree[idx] = wtree[idx](wtree[idx*2], wtree[idx*2+1])
            # except ZeroDivisionError:
            #     wtree[idx] = 100000
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
    w = 0
    while n < iterations:
        tree = function_generator()
        w +=1
        if "x" in tree:
            if get_tree_depth(tree) > 2:
                #check fitness
                #if fitness is better than prev, best_func = tree
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
                #y_pred = [evaluator(tree, x) for x in x_vals]
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

if __name__ == "__main__":


    #print(tree_editor([0,sin], [2]))
    #random_search(10)
    print(function_generator())
