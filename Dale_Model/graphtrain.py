
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from graphtask import *
from graphmeta import nItems, mapping, mappingN, Gedges


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
        print(f'{epoch} {self.loss.data:.4f}')
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
            print(f'{epoch} {self.loss.data:.4f}')
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
        list_len = 4
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