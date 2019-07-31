"""
Author: poiroot
"""
import datetime as dt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from ...dataport.rtpms import RTPMS_OleDB
from ...dataport.lims import Lims
import itertools


class FeatureTransform():

    def __init__(self, config):
        self.author = None
        self.config = config
        self.predict_items = list(config['predict_items'].keys())
        rtpms = create_engine(config['sql_connect']['rtpms'])
        self.rtpms = RTPMS_OleDB(rtpms)
        lims_engine = create_engine(config['sql_connect']['lims'])
        lims_server = config['lims_setting']['history_linked_server']
        lims_table = config['lims_setting']['history_view']
        self.lims = Lims(lims_engine, lims_server, lims_table)
        self.scaler_dict = dict()
        self.prep_steps_dict = dict()
        for key in self.predict_items:
            self.prep_steps_dict[key] = config['predict_items'][key]['prep_steps']

    def _get_data(self):
        """
        Get RTPMS data

        Return type should be pd.DataFrame()
        """
        # TODO implement this method
        return pd.DataFrame()

    def transform(self, time):
        """
        Transform feature

        Return type should be pd.DataFrame()
        """
        # TODO implement this method
        return pd.DataFrame()

    def inverse_transform(self, X):
        columns = X.columns
        for prep_step in self.prep_steps[::-1]:
            X = prep_step.inverse_transform(X)
        return pd.DataFrame(X, columns=columns)

class Utility_Method():

    def __init__(self):  
        self.range_minutes = 10
        self.rtpms_time_step = '00:01:00'

    
    
    def lims_pivot(self, df):
        df.drop_duplicates(['SAMPLED_DATE', 'COMPONENT_NAME'], keep='first', inplace=True)
        df = df.pivot(index='SAMPLED_DATE', columns='COMPONENT_NAME', values='RESULT_VALUE')
        df.index = pd.to_datetime(df.index)
        return df

    # 建立feature
    def create_feature(self, df):

        return df


