# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# get the dimension length of the shape
# for example, shape = [3, 3], the dim_length is 3 * 3 = 9
def get_dim_length(shape):
    if len(shape) != 1:
        dim_length = 1
        for s in shape:
            dim_length *= s
    else:
        dim_length = shape[0]
    return dim_length
