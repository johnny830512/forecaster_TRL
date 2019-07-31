# coding: utf8
import pandas as pd
import numpy as np
import datetime as dt
from .template import FeatureTransform
from .template import Utility_Method
from sqlalchemy import create_engine
from ...dataport.rtpms import RTPMS_OleDB
from ...dataport.lims import Lims

class TLL_viscosity(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[0]
        #self.scaler = self.scaler_dict[self.predict_name]
        self.prep_steps = self.prep_steps_dict[self.predict_name]
        self.u_method = Utility_Method()
        self.rtpms_tags = ['JPC1-FCMA1501.PV','JPC1-FCMA1502.PV','JPC1-VSMA150A2.PV','JPC1-FCT161A3.PV','JPC1-TIK161A3.PV',
            'JPC1-FCAK161A31.PV','JPC1-FCAK161A32.PV','JPC1-FCAK161A33.PV','JPC1-TTAK161A31.PV',
            'JPC1-TTAK161A32.PV','JPC1-TTAK161A33.PV','JPC1-TTAK161A34.PV','JPC1-PCA422E.PV',
            'JPC1-IMK161U.PV','JPC1-AMA162.PV','JPC1-VSMA162A3.PV']

    def _get_data(self, time):
        # overwriting
        start_time = pd.Timestamp(time)

        return self.rtpms.get_rtpms(self.rtpms_tags, start_time, time, "00:01:00")

    def transform(self, time):
        # TODO
        # input time : python datetime object
        # return pandas dataframe
        feature_col = ['input_viscosity','SL_2nd_flow','SL_3rd_flow','CDL_1st_temp',
            'CDL_2nd_temp','TOP_temp','Luwa_amp','dilute_ratio']
        df = self._get_data(time)
           
        df = df.reindex(['JPC1-FCMA1501.PV','JPC1-FCMA1502.PV','JPC1-VSMA150A2.PV',
            'JPC1-FCT161A3.PV','JPC1-TIK161A3.PV','JPC1-FCAK161A31.PV',
            'JPC1-FCAK161A32.PV','JPC1-FCAK161A33.PV','JPC1-TTAK161A31.PV',
            'JPC1-TTAK161A32.PV','JPC1-TTAK161A33.PV','JPC1-TTAK161A34.PV','JPC1-PCA422E.PV',
            'JPC1-IMK161U.PV','JPC1-AMA162.PV','JPC1-VSMA162A3.PV'] , axis = 'columns')
            
        df.columns = ['input_flow1', 'input_flow2', 'input_viscosity', 'dilute_dmf',
            'SL_temp', 'SL_1st_flow', 'SL_2nd_flow', 'SL_3rd_flow', 'CDL_1st_temp',
            'CDL_2nd_temp', 'CDL_3rd_temp', 'TOP_temp', 'vacuum', 'Luwa_amp',
            'Stamo_amp', 'output_viscosity']
        
        # 在下面的 df[feature_col] 會把用不到的欄位過濾掉
        
        df['dilute_ratio'] = df['input_flow1'] / df['dilute_dmf']
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        data_s = self.prep_steps[0].transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        return df
    
class TLL_viscosity_Conv1D(FeatureTransform):

    def __init__(self, config):
        super().__init__(config)
        self.predict_name = self.predict_items[1]
        #self.scaler = self.scaler_dict[self.predict_name]
        self.prep_steps = self.prep_steps_dict[self.predict_name]
        self.u_method = Utility_Method()
        self.rtpms_tags = ['JPC1-FCMA1501.PV','JPC1-VSMA150A2.PV','JPC1-FCT161A3.PV',
            'JPC1-TIK161A3.PV','JPC1-FCAK161A31.PV','JPC1-FCAK161A32.PV','JPC1-FCAK161A33.PV',
            'JPC1-TTAK161A31.PV','JPC1-TTAK161A32.PV','JPC1-TTAK161A33.PV','JPC1-TTAK161A34.PV',
            'JPC1-PCA422E.PV','JPC1-IMK161U.PV','JPC1-AMA162.PV']

    def _get_data(self, time):
        # overwriting
        start_time = pd.Timestamp(time) - dt.timedelta(minutes = 3)

        return self.rtpms.get_rtpms(self.rtpms_tags, start_time, time, "00:01:00")

    def transform(self, time):
        # TODO
        # input time : python datetime object
        # return pandas dataframe
        feature_col = ['input_viscosity', 'SL_temp', 'SL_1st_flow', 'SL_2nd_flow',
           'SL_3rd_flow', 'CDL_1st_temp', 'CDL_2nd_temp', 'CLD_3rd_temp',
           'TOP_temp', 'vacuum', 'Luwa_amp', 'dilute_ratio']
        df = self._get_data(time)
           
        df = df.reindex(['JPC1-FCMA1501.PV','JPC1-VSMA150A2.PV','JPC1-FCT161A3.PV',
            'JPC1-TIK161A3.PV','JPC1-FCAK161A31.PV','JPC1-FCAK161A32.PV','JPC1-FCAK161A33.PV',
            'JPC1-TTAK161A31.PV','JPC1-TTAK161A32.PV','JPC1-TTAK161A33.PV','JPC1-TTAK161A34.PV',
            'JPC1-PCA422E.PV','JPC1-IMK161U.PV','JPC1-AMA162.PV'] , axis = 'columns')
            
        df.columns = ['input_flow','input_viscosity','dilute_dmf','SL_temp','SL_1st_flow',
                      'SL_2nd_flow','SL_3rd_flow','CDL_1st_temp','CDL_2nd_temp','CLD_3rd_temp',
                      'TOP_temp','vacuum','Luwa_amp','Stamo_amp']
        
        # 在下面的 df[feature_col] 會把用不到的欄位過濾掉
        
        
        df['dilute_ratio'] = df['input_flow'] / df['dilute_dmf']
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)  
        
        
        data_s = self.prep_steps[0].transform(df[feature_col])
        df = pd.DataFrame(data=data_s, columns=feature_col)
        
        
        #A => A(t-0),A(t-1),A(t-2),A(t-3)
        input_seq = 4
        df_list = list()
        names = list()

        columns = df.columns
        for i in range(0, input_seq):
            df_list.append(df.shift(i))
            names += ['{0}(t-{1})'.format(c, i) for c in columns]

        df = pd.concat(df_list, axis=1)
        df.dropna(inplace = True)
        df.columns = names
        
        print(df)
        #df = pd.DataFrame(data=data_s, columns=feature_col)
        return df
