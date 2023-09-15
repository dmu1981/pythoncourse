import torch

# Tensoren befinden sich immer auf einem bestimmten device
# Dies ist standarmäßig die CPU, die Daten befinden sich also im RAM
A = torch.rand(5,3)
print(A.device)

# Tensoren können zwischen verschiedenen Devices verschoben werden
# Dadurch werden die Daten physikalisch kopiert. Das ist aufwendig!
B = A.to("cuda")
print(B.device)
