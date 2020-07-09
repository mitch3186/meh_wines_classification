import pickle
import pandas as pd
import numpy as np

# read in the model
my_model = pickle.load(open("finalized_rf_model.pickle","rb"))

# create a function to take in user-entered amounts and apply the model
def good_or_meh_wine(amounts, model=my_model):
    
    # fixed_acidity = attributes_float[0]
    # volatile_acidity = attributes_float[1]
    # residual_sugar = attributes_float[2]
    # chlorides = attributes_float[3]
    # sulphates = attributes_float[4]
    # alcohol = attributes_float[5]
    # sul_diox_ratio = attributes_float[6]
    # input_w = [[fixed_acidity,volatile_acidity,residual_sugar,chlorides,sulphates,alcohol,sul_diox_ratio]]
    #inputs into the model
    # attributes_float.reshape(1,-1)
    input_df = [amounts]
    
    # make a prediction
    prediction = my_model.predict(input_df)[0]

    # return a message
    message_array = ["That juice is loose, it's good!","Meh, hope it was on sale, but please pair it with strong flavors."]

    return message_array[prediction]
