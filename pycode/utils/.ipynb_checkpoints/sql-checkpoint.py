import datetime as dt
import pandas as pd
import logging
from collections import Iterable
from sqlalchemy import *
from sqlalchemy.orm import relationship, create_session, mapper, scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.mssql import BIT, UNIQUEIDENTIFIER
from pycode.utils.load import load_config, load_data

logger = logging.getLogger(__name__)

class TellerSQL:

    def __init__(self, config, engine):
        self.config = config
        self.engine = engine
        Session = scoped_session(sessionmaker(bind=self.engine))
        self.session = Session()
        self.metadata = MetaData(bind=self.engine)
        self.table_list = list()
        self.metadata.reflect(engine)
        for t in self.metadata.sorted_tables:
            self.table_list.append(t.name)

    def create_model_table(self, table_list):

        if len(table_list) == 0:
            return

        model_dic = self.config['predict_items']

        for table in table_list:
            predict_col = list()
            coef_col = list()

            # create predict columns
            if 'Model_' + table not in self.table_list:
                for name in model_dic[table]['algo_name']:
                    predict_col.append(Column('predict_' + name.split('.')[0], FLOAT))
                    coef_col.append(Column('coef_' + name.split('.')[0], FLOAT))

                custom_col = predict_col + coef_col

                Table('Model_' + table, self.metadata,
                      Column('id', Integer, primary_key=True, autoincrement=True),
                      Column('time', DateTime), Column('timestamp', Integer),
                      Column('dqix', FLOAT), Column('ma_dist', FLOAT), Column('mae', FLOAT),
                      Column('conf_idx', FLOAT), *custom_col, Column('predict', FLOAT), Column('intercept', FLOAT))

            if 'Model_' + table + '_var' not in self.table_list:
                Table('Model_' + table + '_var', self.metadata,
                      Column('id', Integer, primary_key=True), Column('item', NVARCHAR(500), primary_key=True),
                      Column('time', DateTime), Column('timestamp', Integer), Column('value', FLOAT))

            if 'Model_' + table + '_mdist' not in self.table_list:
                Table('Model_' + table + '_mdist', self.metadata,
                      Column('id', Integer, primary_key=True), Column('item', NVARCHAR(500), primary_key=True),
                      Column('time', DateTime), Column('timestamp', Integer), Column('value', FLOAT))

        self.metadata.create_all()
        logger.info('Create model table')

    def check_model_table(self):

        model_list = list()
        for key in self.config['predict_items'].keys():
            model_list.append(key)

        lacking_list = list()
        for model in model_list:
            table_name = 'Model_' + model
            if table_name not in self.table_list:
                lacking_list.append(model)
        return lacking_list

    def save_result(self, teller):

        class Model(object):
            pass

        main_table_name = 'Model_' + teller.name
        var_table_name = 'Model_' + teller.name + '_var'
        mdist_table_name = 'Model_' + teller.name + '_mdist'

        model_table = Table(main_table_name, self.metadata, autoload=True, autoload_with=self.engine)
        mapper(Model, model_table)

        m = Model()
        m.time = teller.time
        m.timestamp = int((m.time - dt.timedelta(hours=8)
                        - dt.datetime(1970, 1, 1)).total_seconds())

        m.ma_dist = teller.MD
        m.dqix = teller.DQIX
        m.conf_idx = teller.RI
        m.intercept = teller.intercept
        m.mae = teller.mae
        m.predict = teller.predict

        for c in teller.coefs.keys():
            attr_name = 'coef_' + c
            setattr(m, attr_name, teller.coefs[c])

        for p in teller.predicts.keys():
            attr_name = 'predict_' + p
            setattr(m, attr_name, teller.predicts[p])

        logger.info('Save predict result')

        try:
            self.session.add(m)
            self.session.flush()
            main_id = m.id
            #save item
            self._save_result_item_(var_table_name, teller.X, main_id, teller.time, m.timestamp)
            self._save_result_item_(mdist_table_name, teller.contribute, main_id, teller.time, m.timestamp)

            self.session.commit()
            self.session.close()
        except Exception as e:
            logger.warning(str(e))
            self.session.rollback()

    def _save_result_item_(self, table_name, data, main_id, time, timestamp):

        class Model(object):
            pass

        model_table = Table(table_name, self.metadata, autoload=True, autoload_with=self.engine)
        mapper(Model, model_table)

        bulk_list = list()

        for col in data.columns:
            m = Model()
            m.id = main_id
            m.time = time
            m.timestamp = timestamp
            m.item = col
            m.value = data[col].values[0]
            bulk_list.append(m)

        self.session.add_all(bulk_list)

    def table_mapper(self, table_name):

        class TableObj(object):
            pass

        table = Table(table_name, self.metadata, autoload=True, autoload_with=self.engine)
        mapper(TableObj, table)
        return TableObj

    def query_to_dict(self, result):
        data = list()
        for row in result:
            record = row.__dict__
            del record['_sa_instance_state']
            data.append(record)
        return data

    def get_model_result(self, model_name, start_time, end_time):
        try:
            model = self.table_mapper('Model_' + model_name)
            result = self.session.query(model).filter(model.time.between(start_time, end_time))
            data = self.query_to_dict(result)
            df = pd.DataFrame.from_records(data)
            logger.info("Get predict result. model: {0}, start_time: {1}, end_time: {2}".format(model_name, start_time, end_time))
        except Exception as e:
            logger.warning(str(e))
            self.session.rollback()
        return df

