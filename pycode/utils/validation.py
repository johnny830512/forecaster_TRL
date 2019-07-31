"""
Author: poiroot
"""
from .load import load_data, load_model
import numpy as np
import pandas as pd
import os

def check_resources(config):
    """
    Check the resources for FortuneTeller class

    Parameters
    ----------
    config : dict
        The dictionary read from yaml config file

    Returns
    -------
    output : dict
        The resources for FortuneTeller class
    """
    output = config
    for element in config:
        if element == 'predict_items':
            for item in output[element].keys():
                output[element][item]["scaler"] = load_model(config[element][item]["prep_dir_path"] + config[element][item]["scaler"])
                prep_steps = list()
                for name in config[element][item]["prep_name"]:
                    prep_steps.append(load_model(config[element][item]["prep_dir_path"] + name))
                output[element][item]["prep_steps"] = prep_steps

    for element in config:
        if element == 'predict_items':
            for item in output[element].keys():
                output[element][item]["train"] = load_data(config[element][item]["data_dir_path"] + config[element][item]["data_name"], config[element][item]["data_target"])
                output[element][item]["models"] = list()
                model_filepaths = list()
                for name in config[element][item]["algo_name"]:
                    model_filepaths.append(config[element][item]["algo_dir_path"] + name)
                for model_filepath in model_filepaths:
                    output[element][item]["models"].append(load_model(model_filepath, output, item))
                output[element][item]["mean"] = np.mean(output[element][item]["train"].data, axis=0)
                output[element][item]["std"] = np.std(output[element][item]["train"].data, axis=0, ddof=0)
                output[element][item]["cov"] = np.cov(output[element][item]["train"].data, rowvar=0)
    return output
