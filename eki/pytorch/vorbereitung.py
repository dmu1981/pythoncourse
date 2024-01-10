import math
from torch.utils.tensorboard import SummaryWriter
import time
writer = SummaryWriter()

for degrees in range(360):
    writer.add_scalar("trigonometry/sine", math.sin(degrees * math.pi / 180.0), degrees)
    writer.add_scalar("trigonometry/cosine", math.cos(degrees * math.pi / 180.0), degrees)
    print(degrees)
    time.sleep(1)
