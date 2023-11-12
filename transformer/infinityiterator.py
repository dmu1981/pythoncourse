class InfinityIterator:
  def __init__(self, data):
    self.data = data
    self.iter = data.__iter__()

  def __next__(self):
    try:
      return self.iter.__next__()
    except StopIteration:
      self.iter = self.data.__iter__()
      return self.iter.__next__()


