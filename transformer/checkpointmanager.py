import os
import random

class CheckpointManager:
  def __init__(self, prefix, root="./", suffix=".pt", higher_is_better=True):
    if not prefix.endswith("_"):
      prefix += "_"

    self.higher_is_better = higher_is_better
    self.prefix = prefix
    self.suffix = suffix
    self.root = root
    pass

  def get_checkpoints_list(self):
    chkpts = []
    for _, _, files in os.walk(self.root):
      for file in files:
        if file.endswith(self.suffix) and file.startswith(self.prefix):
          clean = file[len(self.prefix):-len(self.suffix)]
          chkpts.append((int(clean), file))
    
    return chkpts
  
  def get_best_checkpoint(self):
    chkpts = self.get_checkpoints_list()    
    chkpts.sort(key=lambda x: x[0], reverse=self.higher_is_better)
    return chkpts[0][1]
  
  def clean_checkpoints(self, keep_best_n=5):
    chkpts = self.get_checkpoints_list()    
    chkpts.sort(key=lambda x: x[0], reverse=not self.higher_is_better)
    chkpts = chkpts[:-(keep_best_n-1)]
    for chkpt in chkpts:
      os.remove(chkpt[1])

  def new_checkpoint_file(self, metric, clean_as_well=False):
    metric = f"{metric:.3f}"
    metric = ''.join([c for c in metric if c.isalnum()])
    fname = self.prefix + metric + self.suffix
    fname = os.path.join(self.root, fname)

    if clean_as_well:
      self.clean_checkpoints()

    return fname

if __name__ == "__main__":
  chkpt = CheckpointManager("swipes", higher_is_better=True)
  chkpt.clean_checkpoints(keep_best_n=2)
  best = chkpt.get_best_checkpoint()
  print(best)

  for i in range(10):
    fname = chkpt.new_checkpoint_file(random.uniform(0, 100))
    with open(fname, "wt") as f:
      f.write("hallo")