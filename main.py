from statistics import mean
import numpy as np
import random
import matplotlib.pyplot as plt 
import math
import time
from sklearn.metrics import mean_squared_error
import pickle

def read_data():
  f = open("data2022_Platinum.txt", "r")
  data = []
  while(True):
      line = f.readline()
      if not line:
          break
      line = [float(x) for x in line.strip().split(",")]
      data.append(tuple(line))
  f.close
  return(data)

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

def MSE(y1, y2):
    return(np.sqrt(sum([(y1[i]-y2[i])**2 for i in range(len(y1))])/len(y1)))
    #print(y1, y2)
    #print(max(y1), max(y2))
    #return(mean_squared_error(y1,y2))

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

def random_search(iterations):
    '''returns all fitness vals down some number of iterations'''
    dat = read_data()
    best_fitness = 10000000
    n = 0
    all_fitnesses = []
    while n < iterations:
        tree = fgen(5)
        try:
            fitness = get_fitness(tree, dat)
            all_fitnesses.append(fitness)
        except ZeroDivisionError:
            continue
        if fitness < best_fitness:
            best_fitness = fitness
        n+=1
    return(all_fitnesses)

def make_child(end=False, uni=False, div_r=False):
    if end:
        choices = ["x","num"]
        child = random.choices(choices, [1,1])[0]
        child = random.uniform(-10,10) if child == "num" else child
    elif uni:
        ops = ["+", "-", "*", "/", "sin", "cos"]
        choices = ["+", "-", "*", "/","x"]
        op_functions = {"+": add, "-": subtract, "*": multiply, "/": divide, "sin": sin, "cos": cos}
        child = random.choices(choices, k=1)[0]
        if child in ops:
            child = op_functions[child]

    elif div_r:
        ops = ["+", "-", "*", "/", "sin", "cos"]
        ops_and_vals = ops + ['num']
        weights = [1,1,1,1,1,1] + [2]
        op_functions = {"+": add, "-": subtract, "*": multiply, "/": divide, "sin": sin, "cos": cos}
        child = random.choices(ops_and_vals, [i for i in weights], k=1)[0]
        if child in ops:
            child = op_functions[child]
        elif child == "num":
            child = random.uniform(-10,10)
        else:
            pass        
        pass
    else:
        ops = ["+", "-", "*", "/", "sin", "cos"]
        ops_and_vals = ops + ['num'] + ['x']
        weights = [1,1,1,1,1,1] + [1] + [1]
        op_functions = {"+": add, "-": subtract, "*": multiply, "/": divide, "sin": sin, "cos": cos}
        child = random.choices(ops_and_vals, [i for i in weights], k=1)[0]
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
        chosen_depth = random.choice([i for i in range(2,max_depth)])
        last_gen = 2**(chosen_depth-1)
        tree = [None for i in range(2**(chosen_depth+1))]
        ops = ["+", "-", "*", "/", "sin", "cos"]
        op_functions = {"+": add, "-": subtract, "*": multiply, "/": divide, "sin": sin, "cos": cos}
        tree[1] = op_functions[random.choice(ops)]
        for idx in range(1, len(tree)):
            node = tree[idx]
            if callable(node):
                if idx >= last_gen:
                    if node == sin or node == cos:
                        tree[idx*2] = make_child(end=True, uni=True)
                    elif node == divide:
                        tree[idx*2] = make_child(True)
                        tree[idx*2+1] = make_child(True,div_r=True)
                    else:
                        tree[idx*2] = make_child(True)
                        tree[idx*2+1] = make_child(True)
                else:
                    if node == sin or node == cos:
                        tree[idx*2] = make_child(uni=True)
                    elif node == divide:
                        tree[idx*2] = make_child(True)
                        tree[idx*2+1] = make_child(True,div_r=True)
                    else:
                        tree[idx*2] = make_child()
                        tree[idx*2+1] = make_child()
        if "x" in tree:
            try:
                dummy = evaluator(tree, 0)
                if dummy == np.inf:
                    done = False
                elif tree == [None, subtract, 'x', 'x']:
                
                    done = False
                else:
                    done = True
            except ZeroDivisionError:
                done = False
            except ValueError:
                done = False
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
    if y_vals ==[]:
        raise ValueError
    return(MSE(y_vals, y_pred) + .0005*len(indiv))




def GP_trunc(n_gen, init_population, dataset, c_rate, mut_rate):

    initial_population = [fgen(5) for i in range(init_population)]
    fitnesses = [get_fitness(i,dataset) for i in initial_population]
    print(sorted(fitnesses))
    best_fits = []
    pool = []
    pool_size = 0
    all_divs = []
    for i in range(n_gen):
        print("Generation: ", i+1)
        initial_population = [initial_population[i] for i in np.argsort(fitnesses)]
        initial_population = initial_population[:len(initial_population)-5]
        initial_population = initial_population[:int(.5*len(initial_population))] #selection
        fitnesses = sorted(fitnesses)
        fitnesses = fitnesses[:len(fitnesses)-5]
        fitnesses = fitnesses[:int(.5*len(fitnesses))]
        
        while pool_size < 2*len(initial_population):
            couple = np.random.choice([i for i in range(len(initial_population))], size=2, replace=False)
            couple = [initial_population[couple[0]], initial_population[couple[1]]]
            if random.choices([True, False], weights=[c_rate,1-c_rate], k=1)[0]:
                print(len(couple[0]), len(couple[1]))
                children = cross_parents(couple)
                children = [mutator_aggressive(i, mut_rate) for i in children] #mutate
            else:
                children = [fgen(5), fgen(5)]
            #print("odsoijd", children)
            [pool.append(i) for i in children]
            pool_size +=2
        #print("======",len(pool))
        #print("dsklndj",len(initial_population))
        div = []
        for i in pool:
            div.append(sum([get_distance(i,j)**2 for j in initial_population])/len(initial_population))
        all_divs.append(np.mean(div))
        
                
        fitnesses = fitnesses[:5] + [get_fitness(individual, dataset) for individual in pool]
        #print(sorted(fitnesses))
        initial_population = initial_population[:5] + pool 
        pool = []
        pool_size = 0 
        best_fits.append(min(fitnesses))
    
    final_population = [initial_population[i] for i in np.argsort(fitnesses)]
    return(final_population,best_fits,all_divs)


def get_distance(indiv1, indiv2):
    test_nums = [0,1,5,7.5,10,15]
    #print(indiv1,indiv2)
    try:
        y1s = [evaluator(indiv1,i) for i in test_nums]
        y2s = [evaluator(indiv2,i) for i in test_nums]
    except ZeroDivisionError:
        print(indiv1)
        print(indiv2)
        raise ValueError

    return(np.sqrt( sum([(y1s[i]-y2s[i])**2 for i in range(len(test_nums))])/len(y1s) ))

def get_distance2(indiv1, indiv2):
    test_nums = [1,2,5,7.5,10,15]

    y1s = [evaluator(indiv1,i) for i in test_nums]
    y2s = [evaluator(indiv2,i) for i in test_nums]
    # print(y1s, y2s)
    # try:
    #     res = mean([(abs(y1s[i]-y2s[i])+1)/(abs(mean([y1s[i], y2s[i]]))+1) for i in range(len(y1s))])

    # except ZeroDivisionError:
    #     res = mean([(abs(y1s[i]-y2s[i])+1)/(abs(mean([y1s[i], y2s[i]]))+1) for i in range(len(y1s))])



    return(mean([(abs(y1s[i]-y2s[i])+1)/(abs(mean([y1s[i], y2s[i]]))+1) for i in range(len(y1s))]))


def GP_crowded(n_gen, init_population, dataset, c_rate, mut_rate):
    initial_population = [fgen(5) for i in range(init_population)]
    fitnesses = [get_fitness(i,dataset) for i in initial_population]
    print(sorted(fitnesses))
    best_fits = []
    pool = []
    pool_size = 0
    all_divs = []


    for gen in range(n_gen):
        print("Generation number: ", gen)
        print("huh",len(initial_population))
        initial_population = [initial_population[i] for i in np.argsort(fitnesses)]
        initial_population = initial_population[:int(.5*len(initial_population))]
        while pool_size < init_population:
            couple = np.random.choice([i for i in range(len(initial_population))], size=2, replace=False)
            #print(couple)
            couple = [initial_population[couple[0]], initial_population[couple[1]]]
            [initial_population.remove(i) for i in couple]    
            children = cross_parents(couple)
            p1_f = get_fitness(couple[0],dataset); p2_f = get_fitness(couple[1],dataset)
            c1_f = get_fitness(children[0],dataset); c2_f = get_fitness(children[1],dataset)
            if get_distance2(couple[0], children[0]) + get_distance2(couple[1], children[1]) < get_distance2(couple[0], children[1]) + get_distance2(couple[1], children[0]):
                if c2_f < p2_f:
                    mut = mutator_aggressive(children[1],mut_rate)
                    if get_fitness(mut,dataset) < c2_f:
                        pool.append(mut)
                    else:
                        pool.append(children[1]) 
                else:
                    pool.append(couple[1])
                if c1_f < p1_f:
                    mut = mutator_aggressive(children[0],mut_rate)
                    if get_fitness(mut,dataset) < c1_f:
                        pool.append(mut)
                    else:
                        pool.append(children[0]) 
                else:
                    pool.append(couple[0])
            else:
                if c2_f < p1_f:
                    mut = mutator_aggressive(children[1],mut_rate)
                    if get_fitness(mut,dataset) < c2_f:
                        pool.append(mut)
                    else:
                        pool.append(children[1]) 
                else:
                    pool.append(couple[0])
                if c1_f < p2_f:
                    mut = mutator_aggressive(children[0],mut_rate)
                    if get_fitness(mut,dataset) < c1_f:
                        pool.append(mut)
                    else:
                        pool.append(children[0]) 
                else:
                    pool.append(couple[1])
            print("kid lengths: ",len(pool[0]),len(pool[1]))
            pool_size = len(pool)
        
        
        # div = []
        # for i in pool:
        #     div.append(sum([get_distance(i,j)**2 for j in initial_population])/len(initial_population))
        # all_divs.append(np.mean(div))

        initial_population = pool
        fitnesses = [get_fitness(i,dataset) for i in initial_population]

        print("gen best = ", min(fitnesses))
        best_fits.append(min(fitnesses))
        pool_size = 0
    
    final_population = [initial_population[i] for i in np.argsort(fitnesses)]
    return(final_population, best_fits)

def det_crowd(next_pop, parents, children, dataset, mut_rate, current_population):
    p1_f = get_fitness(parents[0],dataset)
    p2_f = get_fitness(parents[1],dataset)
    c1_f = get_fitness(children[0],dataset) 
    c2_f = get_fitness(children[1],dataset)
    if get_distance2(parents[0], children[0]) + get_distance2(parents[1], children[1]) < get_distance2(parents[0], children[1]) + get_distance2(parents[1], children[0]):
        if c2_f < p2_f:
            mut = mutator_aggressive(children[1],mut_rate)
            if get_fitness(mut,dataset) < c2_f:
                next_pop.add(tuple(mut))
            else:
                next_pop.add(tuple(children[1])) 
        else:
            next_pop.add(tuple(parents[1]))
        if c1_f < p1_f:
            mut = mutator_aggressive(children[0],mut_rate)
            if get_fitness(mut,dataset) < c1_f:
                next_pop.add(tuple(mut))
            else:
                next_pop.add(tuple(children[0]))
        else:
            next_pop.add(tuple(parents[0]))
    else:
        if c2_f < p1_f:
            mut = mutator_aggressive(children[1],mut_rate)
            if get_fitness(mut,dataset) < c2_f:
                next_pop.add(tuple(mut))
            else:
                next_pop.add(tuple(children[1])) 
        else:
            next_pop.add(tuple(parents[0]))
        if c1_f < p2_f:
            mut = mutator_aggressive(children[0],mut_rate)
            if get_fitness(mut,dataset) < c1_f:
                next_pop.add(tuple(mut))
            else:
                next_pop.add(tuple(children[0])) 
        else:
            next_pop.add(tuple(parents[1]))


def GP_crowded2(n_gen, init_population, dataset, c_rate, mut_rate):
    # initialize starting population
    # calculate the fitnesses for the population and sort population accordingly

    # repeat for each generation
        # initialize set for storing the individuals for the next generation
        # repeat until the population for the next generation is ready
            # select 2 parents from current generation
            # crossover the 2 parents
            # select who moves to the next generation via deterministic crowding
            # add the 2 diverse and fit bois to the next generation's population
        # replace the current generation population with the next generation's population
        # calculate the fitnesses for the population and sort population accordingly
        # check generation diversity
        # store the value of the best fitness of the generation
    
    #return the best individual of the latest generation

    current_pop = [fgen(4) for i in range(init_population)]
    fits = [get_fitness(i,dataset) for i in current_pop]
    # fit_dict = {}
    # for i in range(len(fits)):
    #     fit_dict[i] = fits[i]
    current_pop = [current_pop[i] for i in np.argsort(fits)]
    best_fits = []
    #fitnesses = [get_fitness(i, dataset) for i in initial_population]

    for i in range(n_gen):
        print("Starting generation ", str(i+1),"...")
        #print(fits)
        next_gen = set()
        while len(next_gen) < init_population:
            #print('len =', len(next_gen))
            parents = selection(current_pop, 0.2, dataset)
            children = cross_parents(parents)
            print('children_size= ', len(children[0]),len(children[1]))
            det_crowd(next_gen,parents,children,dataset,mut_rate,current_pop)
            next_gen.add(tuple(fgen(4)))
            next_gen.add(tuple(fgen(4)))

        
        # current_pop = [list(i) for i in next_gen] + current_pop[:5] #elitism
        current_pop = [list(i) for i in next_gen] 
        fits = [get_fitness(i, dataset) for i in current_pop]
        current_pop = [current_pop[i] for i in np.argsort(fits)]
        # current_pop = current_pop[:-5]
        #print(current_pop)
        print("Gen best: ",min(fits))
        best_fits.append(min(fits))
        #diversity check
        #store diversity value
    print(sorted(fits))
    print(current_pop[0])
    
    return(current_pop[0],best_fits)



def selection(population, p_c, dataset, method="rank"):
    if method == "rank":
        probabilities = np.array([((1-p_c)**((i+1)-1))*p_c for i in range(len(population)-1)] + [(1-p_c)**(len(population))])
        probabilities /= probabilities.sum()
    return(np.random.choice(population,size=2, p=probabilities, replace=False))


    
def hill_climber(n_iterations, data):
    '''returns calculated fitnesses across n_iterations'''
    all_bests = []
    current_t = fgen(6)
    current_fit = get_fitness(current_t, data)
    all_bests.append(current_fit)

    next_t = mutator_aggressive(current_t,1)
    next_fit = get_fitness(next_t, data)
    all_bests.append(next_fit)
    
    for i in range(n_iterations):
        print(i)
        if next_fit < current_fit:
            current_t = next_t
            current_fit = next_fit
               
        next_t = mutator_aggressive(current_t, 1)
        next_fit = get_fitness(next_t, data)
        all_bests.append(next_fit)

    return(all_bests)


def mutator(tree, mut_rate):
    mutation = random.choices([True, False], weights=[mut_rate, 1-mut_rate], k=1)[0]
    if not mutation:
        return(tree)
    valid_ids = []
    #print(True)
    for idx, val in enumerate(tree):
        if type(val) == float:
            valid_ids.append(idx)
    new_vals = [random.choice([-tree[i]*1.1,tree[i]*1.1]) for i in valid_ids]

    for i in range(len(valid_ids)):
        tree[valid_ids[i]] = new_vals[i]

    return(tree)

def mutator_aggressive(tree, mut_rate):
    '''aggressive mutation by adding the option of mutating operators'''
    mutation = random.choices([True, False], weights=[mut_rate, 1-mut_rate], k=1)[0]
    
    if not mutation:
        return(tree)
    wtree = tree.copy()
    done = False
    while not done:
        #print(3)
        lucky_node = random.choice([i for i in range(1,len(wtree))]) #choose a lucky node
        if callable(wtree[lucky_node]):
            if wtree[lucky_node] == sin:
                #replace with cosine
                wtree[lucky_node] = cos
                done = True
                
            elif wtree[lucky_node] == cos:
                #replace with sin
                wtree[lucky_node] = sin
                done = True
            else:
                ops = [add, subtract, multiply, divide]
                #op_functions = {"+": add, "-": subtract, "*": multiply, "/": divide}
                ops.remove(wtree[lucky_node])
                wtree[lucky_node] = random.choice(ops)
                done = True
        elif type(wtree[lucky_node]) == float:
            wtree[lucky_node] = float(random.uniform(-10,10))
            done = True
        else:
            continue

        try:
            evaluator(wtree, 0)
            if wtree == [None, subtract, 'x', 'x']:
                done = False
        except ZeroDivisionError:
            done = False
    #print("eee")
    #print(wtree)
    return(wtree)







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
        if (couple[0][left_idx] == None) or (couple[1][right_idx] == None):
            bad = True
            continue
        else:
            bad = False
            l_idx_depth = math.floor(math.log(left_idx)/math.log(2)); r_idx_depth = math.floor(math.log(right_idx)/math.log(2))
            delta = right_idx-left_idx
            left_nodes = [left_idx] + get_subnodes(couple[0], left_idx); right_nodes = [right_idx] + get_subnodes(couple[1], right_idx)
            l_nodes_trans = []; r_nodes_trans = []
            new_r_tree = couple[1].copy(); new_l_tree = couple[0].copy()
            for i in left_nodes:
                new_l_tree[i] = None
                i_depth = math.floor(math.log(i)/math.log(2))
                l_nodes_trans.append(i + delta* 2 **(i_depth -l_idx_depth))
            for j in right_nodes:
                new_r_tree[j] = None
                j_depth = math.floor(math.log(j)/math.log(2))
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

            for i in [new_l_tree, new_r_tree]:
                try:
                    evaluator(i,0)
                    if len(i) < 110 and "x" in i:
                        pass
                    elif i == [None, subtract, 'x', 'x']:
                        
                        bad = True
                    else:
                        bad = True
                except ZeroDivisionError:
                    bad = True
                    continue

    
    return(new_l_tree, new_r_tree)
    


def plot_soln(tree, dataset):
    x_vals = [tup[0] for tup in dataset]
    y_vals = [tup[1] for tup in dataset]
    

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
    fitness = np.sqrt(sum([(y_vals[x_indices[i]]-y_pred[i])**2 for i in range(len(x_indices))])/len(x_indices)) #with no length loss term


    print(fitness)
    print(tree)


    plt.plot(x_vals, y_vals,"y")
    plt.plot(x_pred, y_pred,"kx", markersize = 2)
    plt.title("Best fit: y = ( sin(1.5001x) * sin( cos( sin(-7.434) ) )"+"\n" +"RMSE: " + str(fitness))
    
    plt.show()  
    

def process_n_plot(data):
    random_res = pickle.load(open( "symreg_results/random_res", "rb"))
    rmhc_res = pickle.load(open( "symreg_results/rmhc_res", "rb"))
    trunc_res = pickle.load(open( "symreg_results/trunc_res", "rb"))
    trunc_soln = pickle.load(open( "symreg_results/trunc_funcs", "rb"))
    d_res = pickle.load(open( "symreg_results/detcrow_res", "rb"))
    d_soln = pickle.load(open( "symreg_results/detcrow_funcs", "rb"))

    ra_res_fixed = []
    for k in range(5):
        fix_r = [random_res[k][0]]
        for i in random_res[k]:
            if i < fix_r[-1]:
                fix_r.append(i)
            else:
                fix_r.append(fix_r[-1])
        ra_res_fixed.append(fix_r)


    rm_res_fixed = []
    for k in range(5):
        fix_r = [rmhc_res[k][0]]
        for i in rmhc_res[k]:
            if i < fix_r[-1]:
                fix_r.append(i)
            else:
                fix_r.append(fix_r[-1])
        rm_res_fixed.append(fix_r)



    GA_trunc_res_fixed = []
    for k in range(5):
        fix_r = [trunc_res[k][0]]
        for i in trunc_res[k]:
            if i < fix_r[-1]:
                fix_r.append(i)
            else:
                fix_r.append(fix_r[-1])
        GA_trunc_res_fixed.append(fix_r)
    

    GA_d_res_fixed = []
    for k in range(5):
        fix_r = [d_res[k][0]]
        for i in d_res[k]:
            if i < fix_r[-1]:
                fix_r.append(i)
            else:
                fix_r.append(fix_r[-1])
        GA_d_res_fixed.append(fix_r)


    ###############
    error_xvals = [i for i in range(0,int(100000/50),int(10000/50))]
    ra_mean1 = [mean([GA_d_res_fixed[0][i], GA_d_res_fixed[1][i], GA_d_res_fixed[2][i],GA_d_res_fixed[3][i],GA_d_res_fixed[4][i]]) for i in range(2001)]
    ra_mean2 = [mean([GA_d_res_fixed[0][i], GA_d_res_fixed[1][i], GA_d_res_fixed[2][i],GA_d_res_fixed[3][i],GA_d_res_fixed[4][i]]) for i in error_xvals]
    ra_sem = [np.std([GA_d_res_fixed[0][i], GA_d_res_fixed[1][i], GA_d_res_fixed[2][i],GA_d_res_fixed[3][i],GA_d_res_fixed[4][i]])/np.sqrt(5) for i in error_xvals]
    plt.plot([i*50 for i in range(2001)], ra_mean1, "m")
    plt.errorbar([i*50 for i in error_xvals],ra_mean2,yerr = ra_sem, fmt = 'o', ecolor = "black", elinewidth=2, capsize=2, markersize=5,markeredgecolor="black",markerfacecolor="black")
    plt.xlabel('Number of Evaluations')
    plt.ylabel('RMSE')

    

    ###################
    
    error_xvals = [i for i in range(0,100000,10000)]
    ra_mean1 = [mean([rm_res_fixed[0][i], rm_res_fixed[1][i], rm_res_fixed[2][i],rm_res_fixed[3][i],rm_res_fixed[4][i]]) for i in range(100001)]
    ra_mean2 = [mean([rm_res_fixed[0][i], rm_res_fixed[1][i], rm_res_fixed[2][i],rm_res_fixed[3][i],rm_res_fixed[4][i]]) for i in error_xvals]
    ra_sem = [np.std([rm_res_fixed[0][i], rm_res_fixed[1][i], rm_res_fixed[2][i],rm_res_fixed[3][i],rm_res_fixed[4][i]])/np.sqrt(5) for i in error_xvals]
    plt.plot([i for i in range(100001)], ra_mean1, "g")
    plt.errorbar(error_xvals,ra_mean2,yerr = ra_sem, fmt = 'o', ecolor = "black", elinewidth=2, capsize=2, markersize=5,markeredgecolor="black",markerfacecolor="black")
    plt.xlabel('Number of Evaluations')
    plt.ylabel('RMSE')
    plt.yscale("log")
    ###############
    error_xvals = [i for i in range(0,int(100000/50),int(10000/50))]
    ra_mean1 = [mean([GA_trunc_res_fixed[0][i], GA_trunc_res_fixed[1][i], GA_trunc_res_fixed[2][i],GA_trunc_res_fixed[3][i],GA_trunc_res_fixed[4][i]]) for i in range(2001)]
    ra_mean2 = [mean([GA_trunc_res_fixed[0][i], GA_trunc_res_fixed[1][i], GA_trunc_res_fixed[2][i],GA_trunc_res_fixed[3][i],GA_trunc_res_fixed[4][i]]) for i in error_xvals]
    ra_sem = [np.std([GA_trunc_res_fixed[0][i], GA_trunc_res_fixed[1][i], GA_trunc_res_fixed[2][i],GA_trunc_res_fixed[3][i],GA_trunc_res_fixed[4][i]])/np.sqrt(5) for i in error_xvals]
    plt.plot([i*50 for i in range(2001)], ra_mean1, "b")
    plt.errorbar([i*50 for i in error_xvals],ra_mean2,yerr = ra_sem, fmt = 'o', ecolor = "black", elinewidth=2, capsize=2, markersize=5,markeredgecolor="black",markerfacecolor="black")
    plt.xlabel('Number of Evaluations')
    plt.ylabel('RMSE')
    ###############
    error_xvals = [i for i in range(0,100000,10000)]
    ra_mean1 = [mean([ra_res_fixed[0][i], ra_res_fixed[1][i], ra_res_fixed[2][i],ra_res_fixed[3][i],ra_res_fixed[4][i]]) for i in range(100001)]
    ra_mean2 = [mean([ra_res_fixed[0][i], ra_res_fixed[1][i], ra_res_fixed[2][i],ra_res_fixed[3][i],ra_res_fixed[4][i]]) for i in error_xvals]
    ra_sem = [np.std([ra_res_fixed[0][i], ra_res_fixed[1][i], ra_res_fixed[2][i],ra_res_fixed[3][i],ra_res_fixed[4][i]])/np.sqrt(5) for i in error_xvals]
    plt.plot([i for i in range(100001)], ra_mean1, "r")
    plt.errorbar(error_xvals,ra_mean2,yerr = ra_sem, fmt = 'o', ecolor = "black", elinewidth=2, capsize=2, markersize=5,markeredgecolor="black",markerfacecolor="black")
    plt.xlabel('Number of Evaluations')
    plt.ylabel('RMSE')
    plt.yscale("log")
    plt.legend(["GP: deterministic crowding","RMHC","GP: truncation selection" ,"Random Search"])
    plt.show()

    



if __name__ == "__main__":
    data = read_data()
    # a = [None, cos, divide, None, 'x', 8.059196563187264]
    # print(get_fitness(a,data))
    # raise ValueError
    #pop = [fgen(5) for i in range(30)]
    #a = selection(pop, .2, data)

    #print(hill_climber(100000,data))
    
    #[print(fgen(5), "\n") for i in range(10)]
 
    # print(a)
    # print(mutator_aggressive(a,1))
    # b = fgen(5)
    # print(b)
    # print("cross", cross_parents([a,b]))

    # raise ValueError
    #a = [None, multiply, 'x', subtract, None, None, multiply, -17.2602, None, None, None, None, -1, "x"]
    #print(get_fitness(a,data))

    # a = [None, multiply, 3, add, None, None, 17, "x"]
    # b =  [None, multiply, "x", subtract, None, None, 17.0000, "x"]
    # c = [None, add, 3, sin, None, None, "x"]
    # print(get_distance2(a,b))
    # print(get_distance(b,c))
    # print(get_distance(a,c))
    #print(mutator_aggressive(a,1))
    # # print(get_fitness(a, data))
    soln,fitnesses= GP_crowded2(3000,30,data, 1,1) #THIS WORKS BUT NEED AT LEAST 2000 EVALS
    print(soln)
    print(fitnesses)
    plot_soln(soln, data)
    plt.plot([i for i in range(len(fitnesses))], fitnesses)
    plt.show()

    # plt.plot([i for i in range(len(divs))], divs)
    # plt.show()
    # # # plt.show()
    # for i in range(10):
    #     f = fgen(5)
    #     y = [evaluator(f,j) for j in [k for k in range(0,10,.1)]]
    #     plt.plot([k for k in range(0,10,.1)], y)
    #     plt.show()
    #[plot_soln(fgen(5),data) for i in range(20)]
    
    #print(evaluator(a,10))
    #print(get_fitness(a,data))








###################################################
    # random_res = []
    # print("STARTING RANDOM SEARCH...")
    # for i in range(5):
    #     random_res.append(random_search(100000))
    # print("FINISHED RANDOM SEARCH")
    # pickle_path = open("random_res", "wb")
    # pickle.dump(random_res, pickle_path)    


    # print("STARTING RMHC...")
    # rmhc_res = []
    # for i in range(5):
    #     rmhc_res.append(100000)
    # pickle_path = open("rmhc_res", "wb")
    # pickle.dump(rmhc_res, pickle_path)
    # print("FINISHED RMHC")


    # print("STARTING GP VIA TRUNCATION SELECTION")
    # GAtrunc_res = []
    # GAtrunc_funcs = []
    # GAtrunc_divs = []
    # for i in range(5):
    #     res = GP_trunc(2000, 50, data, 1, 0.1)
    #     GAtrunc_res.append(res[1])
    #     GAtrunc_funcs.append(res[0][0])
    #     GAtrunc_divs.append(res[2])
    #     print("trial ", i," done.")
    # pickle_path = open("trunc_res", "wb")
    # pickle.dump(GAtrunc_res, pickle_path)
    # pickle_path = open("trunc_funcs", "wb")
    # pickle.dump(GAtrunc_funcs, pickle_path)
    # pickle_path = open("trunc_divs", "wb")
    # pickle.dump(GAtrunc_divs, pickle_path)
    # print("FINSHED GP VIA TRUCNATION SELECTION")



    # print("STARTING GP WITH DETERMINISTIC CROWDING...")
    # GAdet_res = []
    # GAdet_funcs = []
    # for i in range(5):
    #     res = GP_crowded(2000, 50, data, 1, 0.1)
    #     GAdet_res.append(res[1])
    #     GAdet_funcs.append(res[0][0])
    #     print("trial ", i," done.")
    # pickle_path = open("detcrow_res", "wb")
    # pickle.dump(GAdet_res, pickle_path)
    # pickle_path = open("detcrow_funcs", "wb")
    # pickle.dump(GAdet_funcs, pickle_path)
    # print("FINISHED GP WITH DETERMINISTIC CROWDING")
    
