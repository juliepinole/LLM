import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder


def print_model_summary(model):
    def get_num_params(params):
        return sum(p.numel() for p in params)
    
    def get_trainable_params(params):
        return sum(p.numel() for p in params if p.requires_grad)

    print(f"{'Layer (type)':<25} {'Output Shape':<25} {'Param #':<15} {'Trainable':<10}")
    print("=" * 75)
    total_params = 0
    total_trainable = 0
    for name, layer in model.named_modules():
        if len(list(layer.children())) > 0:  # Skip parent layers
            continue
        layer_params = list(layer.parameters())
        num_params = get_num_params(layer_params)
        num_trainable = get_trainable_params(layer_params)
        output_shape = 'N/A'  # Placeholder; actual output shape requires a forward pass
        is_trainable = 'Yes' if num_trainable > 0 else 'No'
        print(f"{name:<25} {output_shape:<25} {num_params:<15} {is_trainable:<10}")
        total_params += num_params
        total_trainable += num_trainable
    
    print("=" * 75)
    print(f"Total params: {total_params}")
    print(f"Trainable params: {total_trainable}")
    print(f"Non-trainable params: {total_params - total_trainable}")



    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
    print("Input shape:", input.shape)
    output = layer(input)
    if isinstance(output, tuple): output = output[0]
    print("Output shape:", output.shape)
    try: reference_output = gpt2_layer(input)
    except: reference_output = gpt2_layer(input, input, input)
    print("Reference output shape:", reference_output.shape, "\n")
    comparison = torch.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f"{comparison.sum()/comparison.numel():.2%} of the values are correct\n")

def isntantiate_layer(cls, gpt2_layer, input, cfg):
    model_tsfm = cls(cfg).to(device)
    model_tsfm.load_state_dict(gpt2_layer.state_dict(), strict=False)
    print("Input shape:", input.shape)
    output = model_tsfm(input)
    if isinstance(output, tuple): output = output[0]
    print("Output shape:", output.shape)
    return output, model_tsfm