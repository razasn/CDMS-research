# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:50:14 2019

@author: 17132
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

with open('data.json') as json_file:  
    json_data = json.load(json_file)


data = pd.DataFrame(json_data['data'])
plt.plot(data['PAS1'][1])
plt.show()