#conding=utf8  
import os 

from count_parameters import count_parameters
g = os.walk("experiments/cifar100/baseline")  

res_dict = {}

for path,dir_list,file_list in g:  
    for dir_name in dir_list:
        sub_folder = os.path.join(path, dir_name)
        if os.path.exists(os.path.join(sub_folder, 'log.txt')):
            # get the results
            _log = os.path.join(sub_folder, 'log.txt')
            _log = open(_log).readlines()[1:]
            if len(_log) >1:
                # get the parameters and flops
                if 'x' in dir_name:
                    arch_name = 'resnet_' + dir_name.split('x')[0].split('_')[1] + 'x'
                    depth = int(dir_name.split('x')[1].split('_')[0])
                else:
                    arch_name = 'resnet'
                    depth = int(dir_name.split('_')[0].replace('resnet', ''))
                flops, parameters = count_parameters(arch_name, depth)

                _top1_list = [float(x.split('\t')[5]) for x in _log] 
                top1 = max(_top1_list)
                name = 'resnet_' + str(depth) + '_' + (arch_name.split('_')[1] if 'x' in arch_name else '1x')
                res_dict[name] = str(top1) + ' ' + str(flops/(1024*1024.0)) + ' ' + str(parameters/(1024.0*1024))

for k, v in res_dict.items():
    print(k + ' ' + v)
