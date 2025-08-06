
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def plot_graphtask(G, mappingN, Gedges=None, font_size=10):
	''' '''
	fig = plt.figure(figsize=(8,3))
	plt.subplot(1,2,1)
	nx.draw(G, with_labels=True,  labels=mappingN, node_size=100, 
			node_color='lightgreen', font_size=font_size, font_color='k')

	if Gedges is not None: 
		plt.subplot(1,2,2)
		plt.imshow(Gedges,cmap='binary')
		plt.title('Edges')

	plt.tight_layout()

def make_inter_trials(edges, nTrials, UNIFORM=False):
	""" Create the interleaved training data
	convert each node to a one hot vector
	and then for each of the edges, we 
	sample each node pair from its one-hot repersentation
	all from a discrete uniform distribution over n trials
	
	ex.
	# nTrials = 176 * 4
	# X,Y = make_inter_trials(edges, nTrials)

	"""
	
	nEdges = len(edges)
	nItems = len(set(edges.flatten()))
	if UNIFORM: 
		edgesSampled = np.random.randint(0, nEdges, nTrials) # random list of edges
	else:
		nReps = nTrials / nEdges # TODO check for unevenness
		l_rep = np.repeat(range(nEdges), nReps)
		edgesSampled = np.random.permutation(l_rep)
		
	trialEdges = edges[edgesSampled] # the repeated edges for each trial (nTrials x 2)

	oneHot = np.eye(nItems)         # one hot template matrix
	X,Y = oneHot[trialEdges[:,0]], oneHot[trialEdges[:,1]]
	
	return X,Y

def search_block_lists(edges, nLists, list_len, niter=50000): 
	"""
	Generates blocked edge lists.
		Randomly generate block lists,
		check if any of the nodes are duplicated,
		if they are, repeat, else, end search.

	ex.
	# nLists = 4
	# list_len = 4

	# blocks = search_block_lists(edges, nLists, list_len)
	# print(blocks) # the edge index for each block
	"""
	nEdges = len(edges)
	fCount = 1000
	for it in range(niter):
		blocks = np.random.choice(range(nEdges), nEdges, replace=False).reshape(list_len,nLists)

		dupCount = 0
		for blockList in range(nLists):
			# Check that all items are unique
			u, c = np.unique(edges[blocks[:,blockList]], return_counts=True)
			dupCount += len(u[c > 1])
			
		if dupCount == 0:
			fCount = dupCount
			if 0: print(it) # print number of iterations to find valid blocking
			break
		else:
			dupCount = 0  
	# Make sure that we are returning a valid search, else, nothing
	if fCount == 0:
		return blocks 
	else:
		return None
	
# Tests
def test_blocks(edges, blocks, list_len):
	""" """
	assert (len(np.unique(edges[blocks[:,0]])) == list_len*2)
	assert (len(np.unique(edges[blocks[:,1]])) == list_len*2)
	assert (len(np.unique(edges[blocks[:,2]])) == list_len*2)
	assert (len(np.unique(edges[blocks[:,3]])) == list_len*2)



def make_block_trials(edges, nTrials, blocks, nItems, nLists, UNIFORM=False):
	"""
	ex.

	# nTrials = 176
	# nLists = 4
	# list_len = 4

	# blocks = search_block_lists(edges, nLists, list_len)
	# test_blocks(edges, blocks, list_len)
	# X_b, Y_b = make_block_trials(edges, nTrials, blocks, nItems, nLists)
	# # np.sum(X_b[0,:,:], axis=0), np.sum(Y_b[0,:,:], axis=0)

	"""
	X_b, Y_b = np.empty((nLists, nTrials, nItems)),np.empty((nLists, nTrials, nItems))
	oneHot = np.eye(nItems)

	for block_list in range(nLists):
		
		if UNIFORM: # Choose edges from uniform distribution
			block_edges_sampled = np.random.choice(blocks[:,block_list], nTrials)
		else: # present shuffled list of perfect numbering
			nReps = nTrials / len(blocks[:,block_list]) # TODO check for unevenness
			bl_rep = np.repeat(blocks[:,block_list], nReps)
			block_edges_sampled = np.random.permutation(bl_rep)
		
		trial_block_edges = edges[block_edges_sampled] # the block list edges
		X,Y = oneHot[trial_block_edges[:,0]], oneHot[trial_block_edges[:,1]]
		X_b[block_list,:,:] = X
		Y_b[block_list,:,:] = Y
		
	return X_b, Y_b



# Task func #

def relative_distance(n_items, model_dists, path_lens, ndistTrials=1000, verbose=False):
	""" 
	model_dists: distance matrix for all model hidden items
	"""
	choice_accs = []
	choice_accs_dist = {0:[],1:[],2:[],3:[],4:[]} # Hardcoded at max PL 4 for now... :(
	for tr in range(ndistTrials):
		# draw 3 random items, without replacement; i2 is reference
		ri = np.random.choice(range(n_items), size=(1,3), replace=False)
		i1,i2,i3 = ri[:,0][0], ri[:,1][0], ri[:,2][0]

		if verbose: print(i1,i2,i3)

		# find the path len for each of the items
		# and the absolute difference between path lens
		d12 = path_lens[i1, i2]
		d32 = path_lens[i3, i2]
		dist_diff = np.abs(d32-d12)

		if verbose: print(tr, 'PL', d12, d32, dist_diff)

		# find which item is "closer" as per shorter 
		# path len 0 = item 1 ; 1 = item 3
		correct_choice = int(np.argmin([d12, d32]))

		# find the models representational similarity for each
		# pair of presented items
		m12 = model_dists[i1, i2]
		m32 = model_dists[i3, i2]

		if verbose: print(tr, 'MD', m12, m32)

		# find which item is "closer" for the model
		# as determined by increased similarity (1 is most similar)
		# 0 = item 1 ; 1 = item 3
		model_choice = int(np.argmax([m12, m32]))

		# assess the model's decision. 1 is correct, 0 is incorrect
		choice_acc = int((correct_choice == model_choice))
		if verbose: print(tr, 'CCMCCA', correct_choice, model_choice, choice_acc)

		choice_accs.append(choice_acc)
		choice_accs_dist[dist_diff].append(choice_acc)
	
	if verbose: print(tr, 'ACC', (np.sum(choice_accs) / ndistTrials) * 100)
	return choice_accs_dist




































