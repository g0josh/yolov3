#!/usr/bin/python

from __future__ import division, print_function

import torch
import torch.nn as nn
import numpy as np

from utils import transform_prediction, get_test_input


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class RouteLayer(nn.Module):
    def __init__(self):
        super(RouteLayer, self).__init__()
        self.layers = []

    def forward(self, x):
        return x


class Darknet(nn.Module):
    """
    A PyTorch implementation of Darknet
    WIP: Contains only layers in YOLOv3
    """
    def __init__(self, cfg_file, color_img = True):
        super(Darknet, self).__init__()
        self.default_image_depth = 3 if color_img else 1
        self.blocks = self.parseCfg(cfg_file)
        self.net_info, self.module_list = self.makeModules(self.blocks)

    def load_weights(self, weightfile):
        
        #Open the weights file
        fp = open(weightfile, "rb")

        #The first 4 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4. Images seen 
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        
        #The rest of the values are the weights
        # Let's load them up
        weights = np.fromfile(fp, dtype = np.float32)
        fp.close()
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                
                conv = model[0]
                
                if (batch_normalize):
                    bn = model[1]
                    
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
                    
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                    
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

    def parseCfg(self, cfg_file):
        """
        Takes a configuration file
        
        Returns a list of blocks. Each blocks describes a block in the neural
        network to be built. Block is represented as a dictionary in the list
        
        """
        lines = []
        file = open(cfg_file, 'r')
        for _line in file:
            lines = lines + [_line.strip()] if len(_line.strip()) > 0 and _line.strip()[0] != '#' else lines
        file.close()

        block = {}
        blocks = []

        for line in lines:
            if line[0] == "[":                      # This marks the start of a new block
                if len(block) > 0:                  # Store the prv block and create a new one
                    blocks.append(dict(block))
                    block = {}
                block["type"] = line[1:-1].strip()     
            else:
                key,value = line.split("=") 
                block[key.strip()] = value.strip()
        blocks.append(dict(block))

        return blocks

    def makeModules(self, blocks):
        """
        Makes only CONVOLUTION, UPSAMPLE, SHORTCUT, ROUTE, YOLO layers as these are the ones in yolov3
        """
        #check the type of block
        #create a new module for the block
        #append to module_list

        net_info = blocks[0]            # Info about the input and pre-processing
        filters_list = []               # list of all the filters (layer depths)
        module_list = nn.ModuleList()
        
        for index, block in enumerate(blocks[1:]):
            module = nn.Sequential()
            if (block["type"] == "net"):
                continue
            elif block['type'] == 'convolutional':
                kernel = int(block["size"])
                filters = int(block['filters'])
                stride = int(block['stride'])
                pad = (kernel - 1) // 2 if 'pad' in block else 0
                bn = True if 'batch_normalize' in block else False
                bias = False if bn else True
                prev_filter = filters_list[-1] if filters_list else self.default_image_depth

                #Add the convolutional layer
                conv = nn.Conv2d(prev_filter, filters, kernel, stride, pad, bias=bias)
                module.add_module("conv_{0}".format(index), conv)
                #Add the Batch Norm Layer
                if bn:
                    batch_norm = nn.BatchNorm2d(filters)
                    module.add_module("batch_norm_{0}".format(index), batch_norm)
                #Activation. Linear or a Leaky ReLU for YOLO
                if block["activation"] == "leaky":
                    activn = nn.LeakyReLU(0.1, inplace = True)
                    module.add_module("leaky_{0}".format(index), activn)
                elif block['activation'] == 'linear':
                    pass
                else:
                    print("Unknown activation please set up\nindex={}\nblock = {}".format(index, block))
                    exit()    

            elif block["type"] == "upsample":
                stride = int(block["stride"])
                upsample = nn.Upsample(scale_factor = stride, mode = "nearest")
                module.add_module("upsample_{}".format(index), upsample)
            elif block['type'] == 'route':
                # Update filters so that convolutional layer can use it in the future
                module.add_module("{}_{}".format(block['type'], index), EmptyLayer())
                layer_indices = [int(x.strip()) for x in block['layers'].split(',')]
                filters = filters_list[layer_indices[0]] if layer_indices[0] > 0 else filters_list[layer_indices[0]+index] 
                print ("layer indices = {}, filters = {}".format(layer_indices, filters))
                for layer_index in layer_indices[1:]:
                    filters += filters_list[layer_index] if layer_index > 0 else filters_list[index+layer_index]
                print ("index = {}, filter = {}".format(index, filters))
            elif block["type"] in ['shortcut', 'yolo']:
                module.add_module("{}_{}".format(block['type'], index), EmptyLayer())

            # elif block["type"] == "yolo":
            #     # anchors = [int(x.strip()) for x in block['anchors'].split(',')]
            #     # anchors = [(anchors[x], anchors[x+1]) for x in range(0, len(anchors), 2)]
            #     # mask = [int(x.strip()) for x in block['mask'].split(',')]
            #     # anchors = [anchors[i] for i in mask]
            #     # detection = DetectionLayer(anchors)
            #     module.add_module("Detection_{}".format(index), EmptyLayer())

            module_list.append(module)
            filters_list.append(filters)
        
        return net_info, module_list

    def forward(self, x, CUDA):
        features_per_layer = []
        detections = torch.FloatTensor()
        for index, block in enumerate(self.blocks[1:]):
            print ("Index = {}, type = {}, input shape = {}".format(index, block['type'], x.shape))
            if block['type'] in ['convolutional', 'upsample']:
                x = self.module_list[index](x)
            elif block['type'] == 'route':
                layer_indices = [int(x.strip()) for x in block['layers'].split(',')]
                x = features_per_layer[layer_indices[0]] if layer_indices[0] > 0 else features_per_layer[index+layer_indices[0]]
                for layer_index in layer_indices[1:]:
                    layer = features_per_layer[layer_index] if layer_index > 0 else features_per_layer[index+layer_index]
                    x = torch.cat((x, layer), 1)
            elif block['type'] == 'shortcut':
                layer_index = int(block['from'].strip()) 
                layer = features_per_layer[layer_index] if layer_index > 0 else features_per_layer[index+layer_index]
                x += layer
            elif block['type'] == 'yolo':
                anchors = [int(y.strip()) for y in block['anchors'].split(',')]
                anchors = [(anchors[y], anchors[y+1]) for y in range(0, len(anchors), 2)]
                mask = [int(y.strip()) for y in block['mask'].split(',')]
                anchors = [anchors[i] for i in mask]
                #Get the input dimensions
                inp_dim = int (self.net_info["height"])
                #Get the number of classes
                num_classes = int (block["classes"])
                detection = transform_prediction(x, inp_dim, anchors, num_classes, CUDA)
                if type(detection) == int:
                    continue
                # If first detection
                if detections.shape == (0,):
                    detections = detection
                else:
                    detections = torch.cat((detections, detection), 1)
            else:
                print ("unknown blocks - {}".format(block['type']))

            features_per_layer.append(x)

        return detections

"""
Testing with a sample image
"""
if __name__ == '__main__':
    inp = get_test_input('images/dog-cycle-car.png')
    print ("Inp shape = {}".format(inp.shape))
    dnn = Darknet('cfg/yolov3.cfg')
    # print ("Module list = {}".format(dnn.module_list))
    dnn.load_weights('yolov3.weights')
    CUDA = torch.cuda.is_available()
    if CUDA:
        dnn.cuda()
        inp = inp.cuda()
    dnn.eval()
    pred = dnn(inp, CUDA)
    print ("prediction = {}\nshape = {}".format(pred, pred.shape))
    

                

