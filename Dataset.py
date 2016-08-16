import numpy as np
import os, glob, threading, random
from enum import Enum

class DatasetState(Enum):
    Null    = 0
    Loaded  = 1
    Loading = 2
    Failed  = 3


class DatasetManager:
    
    def __init__(self,
                 train=None,
                 test =None, 
                 epochs_per_ds=5,
                 target_shape=None):
        
        
        self.fake_data = False
        self.data_path = train
        self.test_path = test
        self.target_shape = target_shape
        self.number_of_datasets = 2
        self.datasets = {}
        self.epochs_per_ds = epochs_per_ds
        
        if train is not None and test is not None:
            self.data_files = glob.glob(self.data_path + "data_*")
            self.test_files = glob.glob(self.test_path + "data_*")
            self.load_dataset(0, async=False)
            self.load_dataset(1, async=True)
        else:
            self.fake_data = True
            self.generate_fake_data(target_shape)
            
        self.active_dataset_index = 0
        
    def generate_fake_data(self, target_shape):
        
        self.datasets['0'] = Dataset(fake=True)
        self.datasets['0'].load(target_shape=target_shape)
        self.datasets['1'] = self.datasets['0']
        
        
        
        
    def get_current_dataset(self) :
        return self.datasets[str(self.active_dataset_index)]
        
    def get_random_dataset_pair(self, type='train'): 
        
        assert type=='train' or type=='test'
        
        filenames = self.data_files if type=='train' else self.test_files
        path      = self.data_path  if type=='train' else self.test_path
        
        data_filename = random.choice(filenames)
        file_id = ((os.path.splitext(os.path.basename(data_filename))[0]).split('_'))[1]
        labels_filename = path + 'labels_' + file_id + '.npy'
        
        return (data_filename, labels_filename)
    
    def load_dataset(self, position, async=False):
        
        # Grab a random patch of data
        (data_filename, labels_filename) = self.get_random_dataset_pair('train')
        
        # Create a new dataset
        self.datasets[str(position)] = Dataset()
        
        # Load it, either sync or async
        return self.datasets[str(position)].load(
            data_filename=data_filename, 
            labels_filename=labels_filename, 
            target_shape=self.target_shape, 
            async=async
        )
        
    def next_dataset(self):
        
        if self.fake_data:
            return self.datasets['0']
        
        # Unload old dataset
        unloaded_ds_index = self.active_dataset_index

        print('Discarding dataset ' + self.datasets[str(unloaded_ds_index)].filename)
        del self.datasets[str(unloaded_ds_index)]

        
        # Shift the current dataset pointer to another dataset that was loaded previously
        self.active_dataset_index += 1
        if self.active_dataset_index > (self.number_of_datasets - 1):
            self.active_dataset_index = 0
        ds = self.datasets[str(self.active_dataset_index)]
        print('Switched to dataset ' + ds.filename)
        
        # Load a newer set in its place Asynchornusly
        self.load_dataset(unloaded_ds_index, async=True)
        
        if ds.state is not DatasetState.Loaded:
            assert self.state is DatasetState.Loading
            ds.loader.join()
        
        return ds

 
    def get_test_dataset(self):
        if self.fake_data:
            return self.datasets['0']
        
        ds = Dataset()
        data_filename, labels_filename = self.get_random_dataset_pair('test')
        return ds.load(
            data_filename=data_filename, 
            labels_filename=labels_filename, 
            target_shape=self.target_shape,
            async=False
        )

    def next_batch(self, batch_size):
        
        ds = self.datasets[str(self.active_dataset_index)]
        if ds._epochs_completed > self.epochs_per_ds:
            ds = self.next_dataset()
        return ds.next_batch(batch_size)
    

class DatasetLoader(threading.Thread):
    
    def __init__(self, parent, labels_filename="", data_filename="", target_shape=None):
        super(DatasetLoader, self).__init__()
        self.labels_filename = labels_filename
        self.data_filename = data_filename
        self.target_shape = target_shape
        
        self.parent = parent
        self.is_done = False
        self.X = None
        self.Y = None
        self.original_X_shape = None
        
    def run(self):
        
        X = np.load(self.data_filename)
        Y = np.load(self.labels_filename)
        
        
        if self.target_shape is not None:
            padding_map = [[0, 0]]
            for i in range(len(self.target_shape)):
                
                padding_size = self.target_shape[i] - X.shape[i + 1]
                
                if padding_size >= 0 :
                    raise("Padding size >= 0 " + str(X.shape) + " -> " + str(self.target_shape))
                
                # If padding size is even then we just put half padding on each side
                # If padding size is odd we add the smaller number before and the larger after
                start = int(padding_size/2)
                end = start + (padding_size%2)
                
                padding_map.append( [start, end] )
                
            # print(padding_map)
            X = np.pad(X, padding_map, mode='constant',constant_values=0)
            # print(X.shape)
        
        self.X = X
        self.Y = Y
        self.original_X_shape = X.shape
        self.is_done = True
        
        self.parent.onDatasetLoaded(self)
    
class Dataset:
    
    def __init__(self, fake=False):
        self.state = DatasetState.Null
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = 0
        self.original_X_shape  = None
        self.loader = None
        self.filename = None
        
        
        self.is_fake = fake
        if fake:
            print('Warning: Using fake data')

        
    def onDatasetLoaded(self, loader):
        
        self.X = loader.X
        self.Y = loader.Y
        self.original_X_shape = loader.original_X_shape
        
        self.X.shape = (self.X.shape[0], -1)
        self._num_examples = self.X.shape[0]
        self.state = DatasetState.Loaded
        print('Loaded dataset ' + loader.data_filename)

    def load_fake_data(self, target_shape):
        
        target_shape = list(target_shape)
        target_shape.insert(0, self._num_examples)
        self._num_examples = 40
        self.original_X_shape = target_shape 
        self.state = DatasetState.Loaded


        dim = 1
        for i in target_shape:
            dim *= i
            
        self.X = np.random.random_sample(size=(self._num_examples, dim))
        self.Y = np.random.random_integers(0, 1, size=(self._num_examples, 2))
        
        print('Loaded fake data')

        return self
        
        
    def load(self, 
             data_filename=None, 
             labels_filename=None, 
             target_shape=None, 
             async=False) :
        
        assert self.state is DatasetState.Null
        
        if self.is_fake:
            return self.load_fake_data(target_shape)
        
        self.loader = DatasetLoader(
            self, 
            data_filename=data_filename, 
            labels_filename=labels_filename, 
            target_shape=target_shape)
        print('Loading dataset ' + self.loader.data_filename + (' (async) ' if async else ' (sync) '))
        self.filename = data_filename
        self.loader.start()
        if async is False:
            self.loader.join()
            return self
        return None

        
    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.X = self.X[perm]
            self.Y = self.Y[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.X[start:end], self.Y[start:end]