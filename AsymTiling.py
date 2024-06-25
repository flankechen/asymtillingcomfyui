import copy
from typing import Any, List, NamedTuple, Optional, Tuple, Union, Callable
import PIL
import torch

from torch import Tensor
from torch import Tensor
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

class AsymTiling:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "active":(["enable", "disable"],),
                "tiling": (["X", "Y"],),
            },
        }
    
    CATEGORY = "conditioning"

    RETURN_TYPES = ("MODEL","VAE")
    FUNCTION = "run"

    def run(self, model, vae, active, tiling):
        if active == "enable":
            if tiling == "X":
                print("asymtil enable x :",)
                model.model.apply(make_circular_asym_x)
                vae.first_stage_model.apply(make_circular_asym_x)

            if tiling == "Y":
                print("asymtil enable y :",)
                model.model.apply(make_circular_asym_y)
                vae.first_stage_model.apply(make_circular_asym_y)
        else:
            print("asymtil restore :",)
            model.model.apply(restore_circular_asym)
            vae.first_stage_model.apply(restore_circular_asym)

        return (model,vae)
        


def make_circular_asym_x(layer):
    # m is each layer now
    if isinstance(layer, torch.nn.Conv2d):
        # m.padding_mode = "circular"
        # print("czz 2 :", layer) 
        layer.padding_modeX = 'circular' 
        layer.padding_modeY = 'constant' 
        layer.paddingX = (layer._reversed_padding_repeated_twice[0], layer._reversed_padding_repeated_twice[1], 0, 0)
        layer.paddingY = (0, 0, layer._reversed_padding_repeated_twice[2], layer._reversed_padding_repeated_twice[3])
        layer._conv_forward = __replacementConv2DConvForward.__get__(layer, Conv2d)

def make_circular_asym_y(layer):
    # m is each layer now
    if isinstance(layer, torch.nn.Conv2d):
        # m.padding_mode = "circular"
        # print("czz 3 :", layer) 
        layer.padding_modeX = 'constant' 
        layer.padding_modeY = 'circular' 
        layer.paddingX = (layer._reversed_padding_repeated_twice[0], layer._reversed_padding_repeated_twice[1], 0, 0)
        layer.paddingY = (0, 0, layer._reversed_padding_repeated_twice[2], layer._reversed_padding_repeated_twice[3])
        layer._conv_forward = __replacementConv2DConvForward.__get__(layer, Conv2d)

def restore_circular_asym(layer):
    if isinstance(layer, torch.nn.Conv2d):
        layer._conv_forward = Conv2d._conv_forward.__get__(layer, Conv2d)

# [Private]
# A replacement for the Conv2d._conv_forward method that pads axes asymmetrically.
# This replacement method performs the same operation (as of torch v1.12.1+cu113), but it pads the X and Y axes separately based on the members
#   padding_modeX (string, either 'circular' or 'constant') 
#   padding_modeY (string, either 'circular' or 'constant')
#   paddingX (tuple, cached copy of _reversed_padding_repeated_twice with the last two values zeroed)
#   paddingY (tuple, cached copy of _reversed_padding_repeated_twice with the first two values zeroed)
def __replacementConv2DConvForward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
    # step = 30 #sample step, just hard code it here
    # if ((self.paddingStartStep < 0 or step >= self.paddingStartStep) and (self.paddingStopStep < 0 or step <= self.paddingStopStep)):
    working = F.pad(input, self.paddingX, mode=self.padding_modeX)
    working = F.pad(working, self.paddingY, mode=self.padding_modeY)
    # else:
    #     working = F.pad(input, self.paddingX, mode='constant')
    #     working = F.pad(working, self.paddingY, mode='constant')
    return F.conv2d(working, weight, bias, self.stride, _pair(0), self.dilation, self.groups)

