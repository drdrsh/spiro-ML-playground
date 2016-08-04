class Dataset:
    
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.X.shape = (self.X.shape[0], -1)
        self._num_examples = X.shape[0]
        self._index_in_epoch = 0
        self._epochs_completed = 0
        
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