import numpy as np
import os, glob, threading, random
from enum import Enum
import queue


class DatasetState(Enum):
    Null = 0
    Loaded = 1
    Loading = 2
    Failed = 3


class DatasetManager:
    def __init__(self,
                 train=None,
                 test=None,
                 validation=None,
                 epochs_per_ds=5,
                 target_shape=None,
                 output_shape=None):

        self.loader_q = queue.Queue()

        self.worker_thread_count = 5
        self.threads = []

        self.fake_data = False
        self.paths = {
            "train": train,
            "test": test,
            "validation": validation
        }

        self.target_shape = target_shape
        self.output_shape = output_shape

        self.number_of_datasets = {
            'train': 4,
            'test': 1,
            'validation': 1
        }

        self.dataset_qs = {}
        self.current_dataset = {}
        self.files = {}
        for i in self.number_of_datasets:
            self.dataset_qs[i] = queue.Queue(self.number_of_datasets[i])
            self.current_dataset[i] = None
            self.files[i] = glob.glob(self.paths[i] + "data_*")

        self.epochs_per_ds = {
            'train': epochs_per_ds,
            'test': None,
            'validation': None
        }

        self.request_dataset('train', self.number_of_datasets['train'])
        self.request_dataset('test', self.number_of_datasets['test'])
        self.request_dataset('validation', self.number_of_datasets['validation'])

        for i in range(self.worker_thread_count):
            t = threading.Thread(target=self.thread_proc_load_dataset)
            t.daemon = True
            t.start()
            self.threads.append(t)

    def thread_proc_load_dataset(self):

        while True:
            source_queue = self.loader_q

            target_type = source_queue.get()
            data_filename, labels_filename = self.get_random_dataset_pair(target_type)

            ds = Dataset.from_file(
                data_filename=data_filename,
                label_filename=labels_filename,
                target_shape=self.target_shape
            )
            target_queue = self.dataset_qs[target_type]
            source_queue.task_done()
            target_queue.put(ds)

    def generate_fake_data(self, target_shape):

        self.datasets['0'] = Dataset(fake=True)
        self.datasets['0'].load(target_shape=target_shape)
        self.datasets['1'] = self.datasets['0']

    def get_current_dataset(self, dataset_type):
        if self.current_dataset[dataset_type] is None:
            self.next_dataset(dataset_type)
        return self.current_dataset[dataset_type]

    def get_random_dataset_pair(self, dataset_type):

        filenames = self.files[dataset_type]
        path = self.paths[dataset_type]

        data_filename = random.choice(filenames)
        file_id = ((os.path.splitext(os.path.basename(data_filename))[0]).split('_'))[1]
        labels_filename = path + 'labels_' + file_id + '.npy'

        return data_filename, labels_filename

    def request_dataset(self, dataset_type, count):
        print("Requesting {0} datasets of type {1}".format(count, dataset_type))
        for i in range(count):
            self.loader_q.put_nowait(dataset_type)

    def next_dataset(self, dataset_type):

        if self.fake_data:
            return self.datasets['0']

        old_ds = self.current_dataset[dataset_type]
        if old_ds is not None:
            print('Discarding dataset ' + old_ds.data_filename)
            del old_ds

        self.current_dataset[dataset_type] = new_ds = self.dataset_qs[dataset_type].get()
        self.dataset_qs[dataset_type].task_done()

        print('Switched to dataset ' + new_ds.data_filename)

        # Request a new dataset to be loaded
        self.request_dataset(dataset_type, 1)

        return new_ds

    def next_batch(self, dataset_type, batch_size):

        ds = self.get_current_dataset(dataset_type)
        if self.epochs_per_ds[dataset_type] is not None and ds.get_epochs_done() > self.epochs_per_ds[dataset_type]:
            ds = self.next_dataset(dataset_type)

        return ds.next_batch(batch_size, output_shape=self.output_shape)


class Dataset:
    @staticmethod
    def from_file(data_filename, label_filename, target_shape):

        ds = Dataset()

        X = np.load(data_filename)
        Y = np.load(label_filename)

        if target_shape is not None:
            padding_map = [[0, 0]]
            for i in range(len(target_shape)):

                padding_size = target_shape[i] - X.shape[i + 1]

                if padding_size < 0:
                    raise ValueError("Padding size < 0 " + str(X.shape) + " -> " + str(target_shape))

                # If padding size is even then we just put half padding on each side
                # If padding size is odd we add the smaller number before and the larger after
                start = int(padding_size / 2)
                end = start + (padding_size % 2)

                padding_map.append([start, end])

            X = np.pad(X, padding_map, mode='constant', constant_values=0)

        ds.X = X
        ds.Y = Y
        ds.original_X_shape = X.shape
        ds.is_done = True
        ds.X.shape = (ds.X.shape[0], -1)
        ds._num_examples = ds.X.shape[0]
        ds.state = DatasetState.Loaded
        ds.data_filename = data_filename
        ds.label_filename = label_filename

        print('Loaded dataset ' + data_filename + ' of the shape ' + str(X.shape))

        return ds

    def get_epochs_done(self):
        return self._epochs_completed

    def __init__(self, fake=False):

        self.state = DatasetState.Null
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = 0
        self.original_X_shape = None
        self.loader = None
        self.filename = None

        self.is_fake = fake
        if fake:
            print('Warning: Using fake data')

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

    def next_batch(self, batch_size, output_shape=None):
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

        data_out = self.X[start:end]
        if output_shape is not None:
            shp = [batch_size] + output_shape
            data_out.shape = shp
        # print(data_out.shape)
        assert len(data_out.shape) > 2
        return data_out, self.Y[start:end]
