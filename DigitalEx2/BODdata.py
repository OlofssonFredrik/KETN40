# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 13:32:35 2023

@author: admin
"""

import numpy as np

BOD = np.array([[1, 2, 3, 4, 5, 7], [8.3, 10.3, 19.0, 16.0, 15.6, 19.8]]).T

BOD2 = np.array(
    [[1, 2, 3, 4, 5, 7, 9, 11], [0.47, 0.74, 1.17, 1.42, 1.60, 1.84, 2.19, 2.17]]
).T

# BOD=[
#     1 8.3
#     2 10.3
#     3 19.0
#     4 16.0
#     5 15.6
#     7 19.8
# ]

# BOD2=[
#   1 0.47
#   2 0.74
#   3 1.17
#   4 1.42
#   5 1.60
#   7 1.84
#   9 2.19
#   11 2.17
# ];