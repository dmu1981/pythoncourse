import torch

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print(f'PyTorch version: {torch.__version__}')
if torch.cuda.is_available():
    print(bcolors.OKGREEN + "  CUDA available" + bcolors.ENDC)
    print(f'Available GPU devices: {torch.cuda.device_count()}')
    print(f'Device Name: {torch.cuda.get_device_name()}')
else:    
    print(bcolors.FAIL + "  CUDA not available" + bcolors.ENDC)

