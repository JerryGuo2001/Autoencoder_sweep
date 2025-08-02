import numpy as np
import torch 
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torchsummary import summary # TODO: This should be a param / file write
#from tqdm import tqdm # TODO: fix verbose training prints to logger or training bar
import networkx as nx
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Now df_frame will have the mean PCA1 and PCA2 for each graph_id
from sklearn.cluster import KMeans
import seaborn as sns
import sys
import torch.nn.init as init
import torch.nn.functional as F
from itertools import permutations

# Get variables from command-line arguments
# partition = int(sys.argv[1])
partition = int(sys.argv[1])
callback= int(sys.argv[2])
arrayid= int(sys.argv[3])

# wd_values = np.logspace(np.log10(1.62e-2), np.log10(1.67e-3), 10)
# lr_values = np.logspace(np.log10(1.0e-1), np.log10(2.78e-5), 10)
iw=''
if partition==1:
    iw='he'
elif partition ==2:
    iw='xavier'
elif partition ==3:
    iw='lecun'
elif partition ==4:
    iw='orthogonal'

wd=1.62e-2
lr=2.78e-4

# Rest of the code

## The old task
# nItems = 12
# mapping = {0:'pot', 1:'mug', 2:'wheel',3:'compass',4:'dice',5:'pan',6:'chair',7:'hammer',
#            8:'clipboard',9:'antenna',10:'triangle',11:'bowl'}
# mappingN = {0:'pot_0', 1:'mug_1', 2:'wheel_2',3:'compass_3',4:'dice_4',5:'pan_5',6:'chair_6',7:'hammer_7',
#            8:'clipboard_8',9:'antenna_9',10:'triangle_10',11:'bowl_11'}

#                   # 0   1.  2.  3.  4.  5.  6.  7.  8.  9.  10. 11. 
# Gedges =  np.array([[0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], # 0
#                    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], # 1
#                    [1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.], # 2
#                    [0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0.], # 3
#                    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], # 4
#                    [0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0.], # 5
#                    [0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.], # 6
#                    [0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 1., 0.], # 7
#                    [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1.], # 8
#                    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.], # 9
#                    [0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1.], # 10
#                    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0.]])# 11

## The new task
nItems = 12
mapping = {0:'skate', 1:'chair', 2:'globe',3:'basket',4:'shades',5:'boat',6:'oven',7:'tree',
           8:'mailbox',9:'fan',10:'pawn',11:'couch'}

mappingN = {0:'skate_0', 1:'chair_1', 2:'globe_2',3:'basket_3',4:'shades_4',5:'boat_5',6:'oven_6',7:'tree_7',
           8:'mailbox_8',9:'fan_9',10:'pawn_10',11:'couch_11'}


# 0   1.  2.  3.  4.  5.  6.  7.  8.  9.  10. 11. 
Gedges =  np.array([
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], 
    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], 
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0], 
    [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0], 
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0], 
    [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
])


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

def parse_dist_accs(frame, dist_id):
    Li, Lb = [], []
    scorei, scoreb = [], []
    for row_idx, row in frame.iterrows():
        if row['task'] == 'I':
            Li.append(row['L2'])
            scorei.append(row['scores'][dist_id])
        elif row['task'] == 'B':
            Lb.append(row['L2'])
            scoreb.append(row['scores'][dist_id])
    
    Li, Lb = np.array(Li), np.array(Lb)
    scorei, scoreb = np.array(scorei), np.array(scoreb)
    return Li, Lb, scorei, scoreb

def plot_results(r_frame):
    dists_l = [1,2,3]
    plt.figure(figsize=(13,3))
    for i in dists_l:
        plt.subplot(1, len(dists_l), i)
        Li, Lb, scorei, scoreb = parse_dist_accs(r_frame, dist_id=i)
        # b inter ,, r blocked
        plt.scatter(Li, scorei, color='teal', alpha=.5)
        plt.scatter(Lb, scoreb, color='r', alpha=.5)

        #find line of best fit
        ai, bi = np.polyfit(Li, scorei, 1)
        ab, bb = np.polyfit(Lb, scoreb, 1)
        plt.plot(Li, ai*Li+bi, color='teal', linewidth=3, label='Intermixed')
        plt.plot(Lb, ab*Lb+bb, color='r', linewidth=3, label='Blocked')
        
        plt.title(f'DistDiff: {i}', size=20)

        if i == 1:
            plt.ylabel('Judgement Accuracy', size=18)
            plt.yticks([0, 25, 50, 75, 100], size=15)
            plt.legend()
        else:
            plt.yticks([])
            

        plt.xlabel('Layer 2 width', size=18)
        plt.ylim(0,100)
        plt.xticks([6, 9, 12, 15, 18], size=15)

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

# def make_inter_trials(edges, nTrials, UNIFORM=False):
# 	""" Create the interleaved training data
# 	convert each node to a one hot vector
# 	and then for each of the edges, we 
# 	sample each node pair from its one-hot repersentation
# 	all from a discrete uniform distribution over n trials
	
# 	ex.
# 	# nTrials = 176 * 4
# 	# X,Y = make_inter_trials(edges, nTrials)

# 	"""
	
# 	nEdges = len(edges)
# 	nItems = len(set(edges.flatten()))
# 	if UNIFORM: 
# 		edgesSampled = np.random.randint(0, nEdges, nTrials) # random list of edges
# 	else:
# 		nReps = nTrials / nEdges # TODO check for unevenness
# 		l_rep = np.repeat(range(nEdges), nReps)
# 		edgesSampled = np.random.permutation(l_rep)
		
# 	trialEdges = edges[edgesSampled] # the repeated edges for each trial (nTrials x 2)

# 	oneHot = np.eye(nItems)         # one hot template matrix
# 	X,Y = oneHot[trialEdges[:,0]], oneHot[trialEdges[:,1]]
	
# 	return X,Y

def make_inter_trials(Gedges, nTrials, UNIFORM=False):
    """
    Create a sequence of (a → b) trials directly from Gedges.
    Returns:
        X: one-hot input of a
        Y: one-hot output of b
        edge_ids: list of (a, b) pairs used in training
    """
    import numpy as np

    G = nx.from_numpy_array(Gedges)
    edges = list(G.edges())  # returns (a, b) tuples
    nEdges = len(edges)
    nItems = Gedges.shape[0]
    oneHot = np.eye(nItems)

    if UNIFORM:
        edge_indices = np.random.randint(0, nEdges, nTrials)
    else:
        nReps = nTrials // nEdges
        l_rep = np.repeat(range(nEdges), nReps)
        edge_indices = np.random.permutation(l_rep)

    X = []
    Y = []
    edge_ids = []

    for idx in edge_indices:
        a, b = edges[idx]

        # randomly flip direction with 50% chance
        if np.random.rand() < 0.5:
            a, b = b, a

        X.append(oneHot[a])
        Y.append(oneHot[b])
        edge_ids.append((a, b))

    X = np.array(X)
    Y = np.array(Y)
    return X, Y, edge_ids




def search_block_lists(edges, nLists, list_len, niter=500000): 
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




# def make_block_trials(edges, nTrials, blocks, nItems, nLists, UNIFORM=False):
# 	"""
# 	ex.

# 	# nTrials = 176
# 	# nLists = 4
# 	# list_len = 4

# 	# blocks = search_block_lists(edges, nLists, list_len)
# 	# test_blocks(edges, blocks, list_len)
# 	# X_b, Y_b = make_block_trials(edges, nTrials, blocks, nItems, nLists)
# 	# # np.sum(X_b[0,:,:], axis=0), np.sum(Y_b[0,:,:], axis=0)

# 	"""
# 	X_b, Y_b = np.empty((nLists, nTrials, nItems)),np.empty((nLists, nTrials, nItems))
# 	oneHot = np.eye(nItems)

# 	for block_list in range(nLists):
		
# 		if UNIFORM: # Choose edges from uniform distribution
# 			block_edges_sampled = np.random.choice(blocks[:,block_list], nTrials)
# 		else: # present shuffled list of perfect numbering
# 			nReps = nTrials / len(blocks[:,block_list]) # TODO check for unevenness
# 			bl_rep = np.repeat(blocks[:,block_list], nReps)
# 			block_edges_sampled = np.random.permutation(bl_rep)
		
# 		trial_block_edges = edges[block_edges_sampled] # the block list edges
# 		X,Y = oneHot[trial_block_edges[:,0]], oneHot[trial_block_edges[:,1]]
# 		X_b[block_list,:,:] = X
# 		Y_b[block_list,:,:] = Y
		
# 	return X_b, Y_b

def make_block_trials(edges, nTrials, blocks, nItems, nLists, UNIFORM=False):
    X_b = np.empty((nLists, nTrials * 2, nItems))
    Y_b = np.empty((nLists, nTrials * 2, nItems))
    edge_ids_b = np.empty((nLists, nTrials, 2), dtype=int)  # New: track edge IDs

    oneHot = np.eye(nItems)

    for block_list in range(nLists):
        if UNIFORM:
            block_edges_sampled = np.random.choice(blocks[:, block_list], nTrials)
        else:
            nReps = nTrials // len(blocks[:, block_list])
            bl_rep = np.repeat(blocks[:, block_list], nReps)
            block_edges_sampled = np.random.permutation(bl_rep)

        trial_block_edges = edges[block_edges_sampled]

        X, Y, edge_ids = [], [], []

        for a, b in trial_block_edges:
            X.append(oneHot[a])
            Y.append(oneHot[a])
            X.append(oneHot[b])
            Y.append(oneHot[b])
            edge_ids.append((a, b))

        X_b[block_list, :, :] = np.array(X)
        Y_b[block_list, :, :] = np.array(Y)
        edge_ids_b[block_list, :, :] = np.array(edge_ids)

    return X_b, Y_b, edge_ids_b




# Task func #

def relative_distance(n_items, model_dists, path_lens, verbose=False):
    """ 
    model_dists: distance matrix for all model hidden items
    path_lens: matrix with path lengths between items
    """
    choice_accs = []
    
    # Use a dictionary with keys from 0 to 4 for valid distances only
    choice_accs_dist = {i: [] for i in range(5)}  # Valid distances: 0 to 4

    # Generate all possible triplets of items (i1, i2, i3)
    all_triplets = list(permutations(range(n_items), 3))  # Generate all unique triplets (i1, i2, i3)

    for triplet in all_triplets:
        i1, i2, i3 = triplet
		
        if verbose: print(i1, i2, i3)

		# Check if the triplet is directly connected
        if (
            path_lens[i1, i2] <= 1 or 
            path_lens[i2, i3] <= 1 or 
            path_lens[i1, i3] <= 1
        ):
            if verbose: print(f"Skipping triplet {triplet} because it's directly connected.")
            continue

        # Calculate path lengths and their absolute difference
        d12 = path_lens[i1, i2]
        d32 = path_lens[i3, i2]
        dist_diff = int(np.abs(d32 - d12))  # Ensure the difference is an integer

        # Skip if the distance difference exceeds 4
        if dist_diff > 4:
            if verbose: print(f"Skipping trial due to large distance difference: {dist_diff}")
            continue

        if verbose: print('PL', d12, d32, dist_diff)

        # Determine the correct choice based on shortest path length
        correct_choice = int(np.argmin([d12, d32]))

        # Retrieve model distances (similarities) between the items
        m12 = model_dists[i1, i2]
        m32 = model_dists[i3, i2]

        if verbose: print('MD', m12, m32)

        # Determine the model's choice based on maximum similarity
        model_choice = int(np.argmax([m12, m32]))

        # Assess the correctness of the model's decision
        choice_acc = int((correct_choice == model_choice))
        if verbose: print('CCMCCA', correct_choice, model_choice, choice_acc)

        # Record accuracy in the appropriate distance bucket
        choice_accs.append(choice_acc)
        choice_accs_dist[dist_diff].append(choice_acc)

    # Print final accuracy if verbose mode is enabled
    if verbose: print('Final ACC', (np.sum(choice_accs) / len(choice_accs)) * 100)

    return choice_accs_dist

class Data:
    def __init__(self, X, Y, edge_ids=None, batch_size=1, datatype=None, shuffle=False, verbose=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.datatype = datatype

        self.dataloader = None
        self.dataloaders = []
        self.blocked_triplet_data = []

        if datatype == 'I':
            self.nblocks = None
            self.data_nsamples = X.shape[0]
            self.data_shape = X.shape[1]
            if verbose: print(f'Data Shape: {self.data_shape} | Batch size {self.batch_size}')
            self.build_dataloader(X, Y)

        elif datatype == 'B':
            self.nblocks = X.shape[0]
            self.data_nsamples = X.shape[1]
            self.data_shape = X.shape[2]
            if verbose: print(f'Data Shape: {self.data_shape} | Batch size {self.batch_size}')
            if edge_ids is None:
                raise ValueError("Must provide edge_ids for blocked triplet training.")
            self.blocked_triplet_data = self.build_blocked_triplet_data(X, Y, edge_ids)
        else:
            raise ValueError('Use either "B" or "I" for datatype')

    def build_dataloader(self, X, Y, out=False): 
        X, Y = np.float32(X), np.float32(Y)
        dataset = TensorDataset(torch.Tensor(X), torch.Tensor(Y))
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, 
                                     shuffle=self.shuffle, drop_last=True)
        if out:
            return self.dataloader

    def build_blocked_triplet_data(self, X, Y, edge_ids):
        """
        Takes block-structured data (X, Y, edge_ids) and returns a list of (Xb, Yb, edge_ids_b) for each block.
        """
        blocks = []
        for block_idx in range(X.shape[0]):
            Xb = X[block_idx]
            Yb = Y[block_idx]
            edge_ids_b = edge_ids[block_idx]
            blocks.append((Xb, Yb, edge_ids_b))
        return blocks



class TrainTorch:

    def __init__(self, model, params):
        self.model = model
        self.num_epochs = params['num_epochs']
        self.learning_rate = params['learning_rate']
        self.weight_decay = params['weight_decay']
        self.device = params['device']
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay #1e-5 #1e-5
        )
        self.is_trained = False

    def train(self, dataloader, verbose_epochs=False, verbose_final=True): 
        ''' '''
        
        loss_store = []
        for epoch in range(self.num_epochs):
            for _, data in enumerate(dataloader):
                X, Y = data
                X, Y = X.to(self.device), Y.to(self.device)

                # forward
                output = self.model(X)
                self.loss = self.criterion(output, Y)
                # backward
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

            if verbose_epochs: print(f'{epoch} {self.loss.data:.4f}')
            loss_store.append(float(f'{self.loss.data:.4f}'))
        self.training_loss = loss_store
        self.is_trained = True

    def train_blocked(self, dataloaders, verbose_blocks=False, verbose_epochs=False, verbose_final=True):
        loss_store = []
        for loaderindex, dataloader in enumerate(dataloaders):
            if verbose_blocks: print(loaderindex)
            for epoch in range(self.num_epochs):
                for _, data in enumerate(dataloader):
                    X, Y = data
                    X, Y = X.to(self.device), Y.to(self.device)

                    # forward
                    output = self.model(X)
                    self.loss = self.criterion(output, Y)
                    # backward
                    self.optimizer.zero_grad()
                    self.loss.backward()
                    self.optimizer.step()

                if verbose_epochs: print(f'{epoch} {self.loss.data:.4f}')
                loss_store.append(float(f'{self.loss.data:.4f}'))
        self.training_loss = loss_store
        self.is_trained = True
    
    def pretrain_reconstruction_only(self, dataloader, verbose_epochs=False):
        self.model.train()
        loss_store = []

        for epoch in range(self.num_epochs):
            for X, _ in dataloader:
                X = X.to(self.device)
                output = self.model(X)
                loss = self.criterion(output, X)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if verbose_epochs:
                print(f"[Pretrain] Epoch {epoch}: MSE = {loss.item():.4f}")
            loss_store.append(loss.item())

        self.training_loss = loss_store
    
    def train_triplet_direct_pairs(self, X, Y, edge_ids, verbose=False):
        self.model.train()
        loss_store = []

        prev_pairs = []

        for epoch in range(self.num_epochs):
            for i in range(0, len(X), 2):  # step through A and B
                a_onehot = X[i]
                b_onehot = X[i + 1]
                a_idx, b_idx = edge_ids[i // 2]

                a = torch.tensor(a_onehot).float().unsqueeze(0).to(self.device)
                b = torch.tensor(b_onehot).float().unsqueeze(0).to(self.device)

                a_embed = self.model(a, encoding=True)
                b_embed = self.model(b, encoding=True)

                if prev_pairs:
                    losses = []
                    for (c_idx, d_idx) in prev_pairs:
                        # (1) ❌ Skip if (a,b) == (c,d) or (b,a) == (c,d)
                        if set([a_idx, b_idx]) == set([c_idx, d_idx]):
                            if verbose:
                                print(f"Skipping full triplet loss for ({a_idx}, {b_idx}) vs ({c_idx}, {d_idx}) — repeated pair")
                            continue

                        # Otherwise continue with individual triplets
                        c = torch.tensor(np.eye(12)[c_idx]).float().unsqueeze(0).to(self.device)
                        d = torch.tensor(np.eye(12)[d_idx]).float().unsqueeze(0).to(self.device)

                        c_embed = self.model(c, encoding=True)
                        d_embed = self.model(d, encoding=True)

                        # (2) ❌ Skip bad triplet combinations
                        if c_idx != a_idx:
                            losses.append(triplet_loss(a_embed, b_embed, c_embed))  # A anchor
                        if d_idx != a_idx:
                            losses.append(triplet_loss(a_embed, b_embed, d_embed))
                        if c_idx != b_idx:
                            losses.append(triplet_loss(b_embed, a_embed, c_embed))  # B anchor
                        if d_idx != b_idx:
                            losses.append(triplet_loss(b_embed, a_embed, d_embed))

                    if losses:
                        tri_loss = sum(losses)
                        self.optimizer.zero_grad()
                        tri_loss.backward()
                        self.optimizer.step()

                        if verbose:
                            print(f"Epoch {epoch} Trial {i // 2}")
                            print(f"  A: {a_idx} and B: {b_idx} | pull closer")
                            for (c_idx, d_idx) in prev_pairs:
                                print(f"  C: {c_idx} and D: {d_idx} | push away")
                            print(f"  Loss: {tri_loss.item():.4f}\n")

                        loss_store.append(tri_loss.item())

                prev_pairs = [(a_idx, b_idx)]  # update memory

        self.training_loss = loss_store






def onehot_to_index(tensor):
    return tensor.argmax(dim=1).tolist()


def get_graph_dataset(edges, sel=''):
    if sel == 'I':
        # Interleaved trials
        X, Y, edge_ids = make_inter_trials(Gedges, nTrials=176*4)
        return X, Y, edge_ids


    elif sel == 'B':
        # Blocked trials
        nTrialsb = 176
        nLists = 4
        list_len = len(edges) // nLists 
        blocks = search_block_lists(edges, nLists, list_len)

        try:
            test_blocks(edges, blocks, list_len)
        except TypeError as e:
            print('Type error, retrying...')
            blocks = search_block_lists(edges, nLists, list_len)
            test_blocks(edges, blocks, list_len)

        Xb, Yb, edge_ids_b = make_block_trials(edges, nTrialsb, blocks, nItems, nLists)
        return Xb, Yb, edge_ids_b

    else:
        raise ValueError('Choose either sel="B" or "I"')



class AE(nn.Module):
	
	def __init__(self, input_shape=100, L1=10, L2=5, n_hidden=5, 
				name='', weight_path=''):
		super(AE, self).__init__()
		self.L1 = L1
		self.L2 = L2

		self.encoder = nn.Sequential(
			nn.Linear(input_shape, self.L1),
			nn.ReLU(True),
			nn.Linear(self.L1, self.L2),
			nn.ReLU(True), 
			nn.Linear(self.L2, n_hidden)
			)

		self.decoder = nn.Sequential(
			nn.Linear(n_hidden, self.L2),
			nn.ReLU(True),
			nn.Linear(self.L2, self.L1),
			nn.ReLU(True), 
			nn.Linear(self.L1, input_shape), 
			nn.Tanh()
			)
		
	def initialize_weights(self, method='xavier'):
		for layer in self.encoder:
			if isinstance(layer, nn.Linear):
				if method == 'xavier':
					init.xavier_uniform_(layer.weight)  # Xavier Initialization
				elif method == 'he':
					init.kaiming_uniform_(layer.weight, nonlinearity='relu')  # He Initialization
				elif method == 'lecun':
					init.normal_(layer.weight, mean=0, std=(1 / layer.weight.size(1)) ** 0.5)  # LeCun Initialization
				elif method == 'orthogonal':
					init.orthogonal_(layer.weight)  # Orthogonal Initialization
				else:
					raise ValueError(f"Unknown initialization method: {method}")
				init.zeros_(layer.bias)  # Initialize biases to 0

		for layer in self.decoder:
			if isinstance(layer, nn.Linear):
				if method == 'xavier':
					init.xavier_uniform_(layer.weight)  # Xavier Initialization
				elif method == 'he':
					init.kaiming_uniform_(layer.weight, nonlinearity='relu')  # He Initialization
				elif method == 'lecun':
					init.normal_(layer.weight, mean=0, std=(1 / layer.weight.size(1)) ** 0.5)  # LeCun Initialization
				elif method == 'orthogonal':
					init.orthogonal_(layer.weight)  # Orthogonal Initialization
				else:
					raise ValueError(f"Unknown initialization method: {method}")
				init.zeros_(layer.bias)  # Initialize biases to 0

	def forward(self, x, encoding=False):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		if encoding: # Return hidden later activations
			return encoded
		return decoded

def get_hidden_activations(model, n_hidden=5, n_items=12, device='cuda'):
    ''' Assumes one hot'''
    test_dat = np.eye(n_items)
    #outarr = np.zeros((n_items, n_items))
    hiddenarr = np.zeros((n_items, n_hidden))
    for i in range(n_items):
        x = torch.Tensor(test_dat[i,:]).to(device)
        #out = model.forward(x, encoding=False).detach()
        hidden = model.forward(x, encoding=True).detach()
        hiddenarr[i, :] = hidden.cuda().numpy() if torch.cuda.is_available() else hidden.cpu().numpy()
    return hiddenarr


#Pretrain Section
def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute triplet loss:
    - anchor, positive, negative: shape (batch_size, hidden_dim)
    """
    d_pos = F.pairwise_distance(anchor, positive, p=2)
    d_neg = F.pairwise_distance(anchor, negative, p=2)
    loss = F.relu(d_pos - d_neg + margin)
    return loss.mean()

def sample_triplets_from_graph(n_items, path_lens, max_dist=4):
    """
    Generate valid triplets (anchor, pos, neg) based on path length difference.
    Returns a list of (a, p, n) indices.
    """
    triplets = []
    for a in range(n_items):
        for p in range(n_items):
            for n in range(n_items):
                if a == p or a == n or p == n:
                    continue
                d_ap = path_lens[a, p]
                d_an = path_lens[a, n]
                if d_ap < d_an and abs(d_ap - d_an) <= max_dist:
                    triplets.append((a, p, n))
    return triplets

# def pretrain_with_triplet(L2=12,n_items=12, hidden_dim=3, method='he', margin=1.0, 
#                           epochs=1, batch_size=8, lr=1e-3, wd=1e-5, device='cpu'):
#     path_lens = nx.floyd_warshall_numpy(nx.from_numpy_array(Gedges))
#     triplets = sample_triplets_from_graph(n_items, path_lens)

#     # Convert to dataset
#     triplet_tensor = torch.tensor(triplets, dtype=torch.long)

#     model = AE(input_shape=n_items, L1=12, L2=L2, n_hidden=hidden_dim).to(device)
#     model.initialize_weights(method)

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

#     model.train()
#     for epoch in range(epochs):
#         # Shuffle and batch
#         perm = torch.randperm(len(triplets))
#         for i in range(0, len(triplets), batch_size):
#             idx = perm[i:i+batch_size]
#             batch = triplet_tensor[idx]
#             a_idx, p_idx, n_idx = batch[:,0], batch[:,1], batch[:,2]

#             X = torch.eye(n_items).to(device)
#             a, p, n = X[a_idx], X[p_idx], X[n_idx]

#             h_a = model(a, encoding=True)
#             h_p = model(p, encoding=True)
#             h_n = model(n, encoding=True)

#             loss = triplet_loss(h_a, h_p, h_n, margin)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
        
#         if epoch % 10 == 0:
#             print(f"Epoch {epoch}: Triplet loss = {loss.item():.4f}")

#     return model.state_dict()


def roll_idx(n):
    '''Create index that moves the first matrix row to the last'''
    idx = list(range(n))[1:n]
    idx.append(0)
    return idx 

def add_one(features=4, samples=5):
    # Check sample equation?
    XX = np.eye(features)
    XY = XX[:, roll_idx(features)]

    XX = np.tile(XX, (samples)).T
    XY = np.tile(XY, (samples)).T
    return XX, XY

def calc_dist(a, b):
    '''
    Requires:
    from scipy.spatial.distance import cdist'''
    return 1 - cdist(a, b, metric='cosine')

def H2I(H):
	""" Convert one-hot back to int """
	return np.where(H)[0][0]

### THE GRAPH TASK 
big_data_frames = []
adj_matrix = Gedges
G = nx.from_numpy_array(Gedges)
edges = np.array(list(G.edges)) 
if 0: plot_graphtask(G, mappingN, Gedges, font_size=10) # TODO: I don't think there's a show call

batman_df=pd.DataFrame()

# Set device and params
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params = {
    'num_epochs': 2,
    'learning_rate': lr,
    'weight_decay': wd, 
    'device': device
}
data_dir = '/Users/jianleguo/Desktop/VAE/original_model/AE_Result' # os.path.join()
save_results = False
save_weights = False
ult_data_frame = pd.DataFrame()
big_data_frames = []
print('ready_to_go')
n_models = 20
hidden_layer_widths = [6,12,18,24,30,36,42,48,54,60,66,68,72,78,84,90,96,102,108,114,120,126,132,138,144]
results = {'name':[], 'path':[], 'task':[], 'L2':[], '1':[], '2':[], '3':[], '4':[],
            'scores':[], 'end_loss':[], 'hidden':[], 'dists':[],'learn_rate':[],'weight_decay':[],'repetition':[],'initial_weight_type':[],'initial_weights':[]}

# Sweet jesus this tripple for loop makes me sad
for dataset_ID in ['I','B']:
    for l2_size in hidden_layer_widths:
        for model_i in range(n_models):
            X, Y, edge_ids = get_graph_dataset(edges, sel=dataset_ID)
            dat = Data(X, Y, edge_ids=edge_ids, batch_size=8, datatype=dataset_ID, shuffle=False)  # shuffle ok for pretrain

            # Set model metadata and create model and trainer
            model_name = f'{dataset_ID}_{l2_size}_{model_i}'
            weight_path = f'{data_dir}/torchweights/{model_name}.pt'
            model = AE(input_shape=dat.data_shape, L1=12, L2=l2_size, n_hidden=12,
                       name=f'{model_name}', weight_path=weight_path).to(device)
            model.initialize_weights(method='he')
            net = TrainTorch(model, params)

            # ✅ PHASE 1 — Pretrain (optional)
            # if dat.datatype == 'I':
            #     net.pretrain_reconstruction_only(dat.dataloader)
            # elif dat.datatype == 'B':
            #     for loader in dat.dataloaders:
            #         net.pretrain_reconstruction_only(loader)

            # ✅ PHASE 2 — Triplet training
            if dat.datatype == 'I':
                net.train_triplet_direct_pairs(X, Y, edge_ids, verbose=False)
            elif dat.datatype == 'B':
                for Xb, Yb, edge_idsb in dat.blocked_triplet_data:
                    net.train_triplet_direct_pairs(Xb, Yb, edge_idsb, verbose=False)

            # save model weights
            if save_weights: torch.save(net.model.state_dict(), weight_path)
            
            # Compute hidden activations and task distances
            hiddenarr = get_hidden_activations(net.model, n_hidden=12, 
                                                n_items=12, device=device)
            model_dists = calc_dist(hiddenarr, hiddenarr)
            path_lens = nx.floyd_warshall_numpy(G)
            trialIter = 500

            # Compute choice task results
            choice_accs_dist = relative_distance(12, model_dists, path_lens, verbose=False)
            dist_pct = {} # TODO: Wrap into function
            for dist, vals in choice_accs_dist.items():
                acc = (np.sum(vals) / len(vals)) * 100
                dist_pct[dist] = acc
                if 0: print(f'{dist}: {acc:.2f}% {len(vals)}')

            # Pack up into dictionary 
            # TODO: this should be an interable
            results['name'].append(model_name)
            results['path'].append(weight_path)
            results['task'].append(dataset_ID)
            results['L2'].append(l2_size)
            results['1'].append(dist_pct[1])
            results['2'].append(dist_pct[2])
            results['3'].append(dist_pct[3])
            results['4'].append(dist_pct[4])
            results['scores'].append(dist_pct)
            results['end_loss'].append(net.training_loss[-1])
            results['hidden'].append(hiddenarr)
            results['dists'].append(model_dists)
            results['learn_rate'].append(lr)
            results['weight_decay'].append(wd)
            results['repetition'].append(partition)
            results['initial_weight_type'].append(iw)
            results['initial_weights'].append('he') 
            print('finish one model')
        print('finished one set')
# Save results
r_frame = pd.DataFrame(results)


r_frame.to_csv(f'output_data/{callback}_{partition}_{arrayid}.csv', index=False)