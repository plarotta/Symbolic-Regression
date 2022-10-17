def tree_editor_deprecated(tree, edits):
    ops = ["+", "-", "*", "/", "sin", "cos"]
    ops_and_vals = ops + ['num'] + ['x']
    weights = [1,1,1,1,1,1] + [1] + [1]
    op_functions = {"+": add, "-": subtract, "*": multiply, "/": divide, "sin": sin, "cos": cos}
    working_tree = tree.copy()
    for idx in edits:
        try:
            if working_tree[idx] == None:
                elem = random.choices(ops_and_vals, [i/1000 for i in weights], k=1)[0]
                if elem in ops:
                    working_tree[idx] = op_functions[elem]
                elif elem == "num":
                    working_tree[idx] = random.uniform(-10,10)
                else:
                    working_tree[idx] = elem
        except IndexError:
            for i in range(idx+1 - len(working_tree)):
                working_tree.append(None) 
            elem = random.choices(ops_and_vals, [i/1000 for i in weights], k=1)[0]
            if elem in ops:
                working_tree[idx] = op_functions[elem]
            elif elem == "num":
                working_tree[idx] = random.uniform(-10,10)
            else:
                working_tree[idx] = elem

    return(working_tree)

def function_generator_deprecated():
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

		tree = tree_editor_deprecated(tree, edits); print(len(tree))
		tree_not_finished = not tree_solver(tree, 1)
	return(tree)