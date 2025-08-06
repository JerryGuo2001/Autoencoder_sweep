import torch

def torch_gpu_info():
    ''' '''
    print(f'Torch available? {torch.cuda.is_available()}') # should be True
    print(f'Device Count: {torch.cuda.device_count()}') # should be > 0
    current_device = torch.cuda.current_device()
    print(f'Current_device: {current_device}')
    print(f'Device name: {torch.cuda.get_device_name(current_device)}')

class Torch_utils:
    def __init__(self) -> None:
        pass