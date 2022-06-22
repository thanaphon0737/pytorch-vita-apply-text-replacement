import pickle
import matplotlib.pyplot as plt 
import numpy
import pandas as pd
with open('./losses/0_e_f_6_01042022-2.pkl', 'rb') as file:
      
    # Call load method to deserialze
    # myvar = pickle.load(file)
    object = pickle.load(file)
    
df = pd.DataFrame(object)
df.to_csv(r'./losses/0_e_f_6_01042022-2.csv')