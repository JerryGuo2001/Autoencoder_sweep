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

# Get variables from command-line arguments
# partition = int(sys.argv[1])
repet = int(sys.argv[1])
callback= int(sys.argv[2])
abc = int(sys.argv[3])

wd=0.01*abc
lr=0.01*abc

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
Gedges =  np.array([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], # 0
                   [1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.], # 1
                   [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], # 2
                   [0., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0.], # 3
                   [0., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0.], # 4
                   [0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0.], # 5
                   [0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0.], # 6
                   [0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.], # 7
                   [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.], # 8
                   [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1.], # 9
                   [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1.], # 10
                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.]])# 11


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
    path_lens: matrix with path lengths between items
    """
    choice_accs = []
    
    # Use a dictionary with keys from 0 to 4 for valid distances only
    choice_accs_dist = {i: [] for i in range(5)}  # Valid distances: 0 to 4

    for tr in range(ndistTrials):
        # Draw 3 random items, without replacement; i2 is reference
        ri = np.random.choice(range(n_items), size=(1, 3), replace=False)
        i1, i2, i3 = ri[:, 0][0], ri[:, 1][0], ri[:, 2][0]

        if verbose: print(i1, i2, i3)

        # Calculate path lengths and their absolute difference
        d12 = path_lens[i1, i2]
        d32 = path_lens[i3, i2]
        dist_diff = int(np.abs(d32 - d12))  # Ensure the difference is an integer

        # Skip if the distance difference exceeds 4
        if dist_diff > 4:
            if verbose: print(f"Skipping trial {tr} due to large distance difference: {dist_diff}")
            continue

        if verbose: print(tr, 'PL', d12, d32, dist_diff)

        # Determine the correct choice based on shortest path length
        correct_choice = int(np.argmin([d12, d32]))

        # Retrieve model distances (similarities) between the items
        m12 = model_dists[i1, i2]
        m32 = model_dists[i3, i2]

        if verbose: print(tr, 'MD', m12, m32)

        # Determine the model's choice based on maximum similarity
        model_choice = int(np.argmax([m12, m32]))

        # Assess the correctness of the model's decision
        choice_acc = int((correct_choice == model_choice))
        if verbose: print(tr, 'CCMCCA', correct_choice, model_choice, choice_acc)

        # Record accuracy in the appropriate distance bucket
        choice_accs.append(choice_acc)
        choice_accs_dist[dist_diff].append(choice_acc)

    # Print final accuracy if verbose mode is enabled
    if verbose: print('Final ACC', (np.sum(choice_accs) / len(choice_accs)) * 100)

    return choice_accs_dist


class Data:
    def __init__(self, X, Y, batch_size=1, datatype=None, shuffle=False, verbose=False):
        
        self.batch_size = batch_size #16 # 2, 4, 
        self.shuffle = shuffle
        self.datatype = datatype

        self.dataloader = None
        self.dataloaders = []

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
            self.build_blocked_dataloaders(X, Y)
        else:
            raise ValueError('Use either "B" or "I" for datatype')

    def build_dataloader(self, X, Y, out=False): 
        ''' '''
        X, Y = np.float32(X), np.float32(Y)
        #X = TensorDataset(torch.from_numpy(X))
        dataset = TensorDataset( torch.Tensor(X), torch.Tensor(Y) )
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, 
                                    shuffle=self.shuffle, drop_last=True)
        
        if out:
            return self.dataloader

    def build_blocked_dataloaders(self, X, Y):
        dataloaders = []
        for block in range(X.shape[0]):
            #print(block)
            Xb, Yb = X[block,:,:], Y[block,:,:]
            #print(Xb.shape)

            dataloader = self.build_dataloader(Xb, Yb, out=True)
            self.dataloaders.append(dataloader)

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

def get_graph_dataset(edges, sel=''):
    # TODO: set all of these to passable param dict
    if sel == 'I':
        # Interleaved trials
        nTrials = 176 * 4
        X, Y = make_inter_trials(edges, nTrials)
        return X,Y
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
        Xb, Yb = make_block_trials(edges, nTrialsb, blocks, nItems, nLists)
        return Xb, Yb
    else:
        raise ValueError('Choose either sel="B" or "I"')

class AE(nn.Module):
	
	def __init__(self, input_shape=100, L1=10, L2=5, n_hidden=3, 
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

	def forward(self, x, encoding=False):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		if encoding: # Return hidden later activations
			return encoded
		return decoded

def get_hidden_activations(model, n_hidden=3, n_items=12, device='cuda'):
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
    'num_epochs': 10,
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
for i in range (10):
    print('we_are_started_at'+f'{i}')
    n_models = 1
    hidden_layer_widths = [6, 9, 12, 15, 18]
    results = {'name':[], 'path':[], 'task':[], 'L2':[], '1':[], '2':[], '3':[], '4':[],
                'scores':[], 'end_loss':[], 'hidden':[], 'dists':[],'learn_rate':[],'weight_decay':[],'repetition':[],'wd_lr_pair':[]}

    # Sweet jesus this tripple for loop makes me sad
    for dataset_ID in ['I', 'B']:
        for l2_size in hidden_layer_widths:
            for model_i in range(n_models):
                # build dataset...
                X, Y = get_graph_dataset(edges, sel=dataset_ID)
                dat = Data(X, Y, batch_size=8, datatype=dataset_ID, shuffle=False)

                # Set model metadata and create model and trainer
                model_name = f'{dataset_ID}_{l2_size}_{model_i}'
                weight_path = f'{data_dir}/torchweights/{model_name}.pt' # os.path.join()
                model = AE(input_shape=dat.data_shape, L1=12, L2=l2_size, n_hidden=3, 
                            name=f'{model_name}', weight_path=weight_path).to(device)
                if 0: summary(model, input_size=(1,20))
                net = TrainTorch(model, params)

                # Train model 
                if dat.datatype == 'I': net.train(dat.dataloader)
                elif dat.datatype == 'B': net.train_blocked(dat.dataloaders)
                else:
                    raise ValueError(f'datatype trainer is {dat.datatype}')

                # save model weights
                if save_weights: torch.save(net.model.state_dict(), weight_path)
                
                # Compute hidden activations and task distances
                hiddenarr = get_hidden_activations(net.model, n_hidden=3, 
                                                    n_items=12, device=device)
                model_dists = calc_dist(hiddenarr, hiddenarr)
                path_lens = nx.floyd_warshall_numpy(G)
                trialIter = 500

                # Compute choice task results
                choice_accs_dist = relative_distance(12, model_dists, path_lens, 
                                                        ndistTrials=trialIter, verbose=False)
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
                results['learn_rate'].append(8.25e-5)
                results['weight_decay'].append(3.78e-5)
                results['repetition'].append(repet)
                results['wd_lr_pair'].append(abc)
        print('model_id:',i)
        
    # Save results
    r_frame = pd.DataFrame(results)
    big_data_frames.append(r_frame)

ult_data_frame = pd.concat(big_data_frames, ignore_index=True)

print(ult_data_frame.shape)

ult_data_frame.to_csv(f'output_data/{callback}_{abc}.csv', index=False)