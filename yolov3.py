#!/usr/bin/python

from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import cv2

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class YoloV3(nn.Module):
    """
    A PyTorch implementation of Yolo v3 which was implemented in Darknet by the Author
    """
    def __init__(self, cfg_file):
        super(YoloV3, self).__init__()
        self.blocks = self.parseCfg(cfg_file)
        self.net_info, self.module_list = self.makeModules(self.blocks)

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
        Makes only CONVOLUTION, UPSAMPLE, RESIDUAL, ROUTE, YOLO layers as these are the ones in yolov3
        """
        #check the type of block
        #create a new module for the block
        #append to module_list

        net_info = blocks[0]            #Info about the input and pre-processing
        prev_filters = 3                # RGB Image
        module_list = nn.ModuleList()
        module = nn.Sequential()
        
        for index, block in enumerate(blocks[1:]):
            if block['type'] == 'convolutional':
                kernel = int(block["size"])
                filters = int(block['filters'])
                stride = int(block['stride'])
                pad = (kernel - 1) // 2 if 'pad' in block else 0
                bn = True if 'batch_normalize' in module else False
                bias = False if bn else True

                #Add the convolutional layer
                conv = nn.Conv2d(prev_filters, filters, kernel, stride, pad, bias)
                module.add_module("conv_{0}".format(index), conv)


                #Add the Batch Norm Layer
                if bn:
                    batch_norm = nn.BatchNorm2d(filters)
                    module.add_module("batch_norm_{0}".format(index), batch_norm)

                #Activation. 
                #It is either Linear or a Leaky ReLU for YOLO
                if block["activation"] == "leaky":
                    activn = nn.LeakyReLU(0.1, inplace = True)
                    module.add_module("leaky_{0}".format(index), activn)
                else:
                    print("Unknown activation please set up manually\nindex={}\nblock = {}".format(index, block))
                    exit()    

            elif block["type"] == "upsample":
                stride = int(block["stride"])
                upsample = nn.Upsample(scale_factor = stride, mode = "bilinear")
                module.add_module("upsample_{}".format(index), upsample)

            elif block["type"] in ["shortcut", "route"]:
                route = EmptyLayer()
                module.add_module("route_{0}".format(index), route)

            elif block["type"] == "yolo":
                anchors = [int(x.strip()) for x in block['anchors'].split(',')]
                anchors = [(anchors[x], anchors[x+1]) for x in range(0, len(anchors), 2)]
                mask = [int(x.strip()) for x in block['mask'].split(',')]
                anchors = [anchors[i] for i in mask]
                detection = DetectionLayer(anchors)
                module.add_module("Detection_{}".format(index), detection)

            module_list.append(module)
        
        return net_info, module_list

    def forward(self, x):
        features_per_layer = []
        detections = []
        for index, block in enumerate(self.blocks[1:]):
            if block['type'] not in ['convolutional', 'upsample']:
                x = self.module_list[index](x)
            elif block['type'] == 'route':
                layer_indices = [int(x.strip()) for x in block['layers'].split(',')]
                x = features_per_layer[layer_indices[0]] if layer_indices[0] > 0 else features_per_layer[index+layer_indices[0]]
                if len(layer_indices) > 1:
                    for layer_index in layer_indices[1:]:
                        layer = features_per_layer[layer_indices[layer_index]] if layer_indices[layer_index] > 0 else features_per_layer[index+layer_indices[layer_index]]
                        x = torch.cat((x, layer), 1)
            elif block['type'] == 'shortcut':
                layer_index = int(block['layer'].strip()) 
                layer = features_per_layer[layer_index] if layer_index > 0 else features_per_layer[index+layer_index]
                x += layer
            elif block['type'] == 'yolo':
                detection = {
                    'prediction': x.data,
                    'anchors':self.module_list[index][0].anchors}            
                detections.append(detection)
            features_per_layer.append(x)


                

