import torch
import argparse
import collections

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def main(args):
    checkpoints = torch.load(args.checkpoints)
    for k in list( checkpoints['state_dict'].keys() ):
        if 'running_var' in k:
            print(k)
            #print(checkpoints['state_dict'][k])
            print(torch.mean(torch.abs(checkpoints['state_dict'][k])))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract the weights of logits layer.")
    parser.add_argument("-c", "--checkpoints", type=str, required=True)
    args = parser.parse_args()
    main(args)
