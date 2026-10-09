import os
import sys
from src.ML_PROJECT import logger
from src.ML_PROJECT.exception import customeException
from dotenv import load_dotenv
import pymysql
import pandas as pd

load_dotenv()

HOST=os.getenv("host")
USER=os.getenv("user")
PASSWORD=os.getenv("password")
DB=os.getenv("db")

def read_sql_data():
    print("ENV DATA :- ", HOST, USER, PASSWORD, DB)
    logger.logging.info("Start a reading Data From MYSQL----")
    try:
        logger.logging.info('Try to Connect My sql')
        connection_sql=pymysql.connect(user=USER,host=HOST,password=PASSWORD,db=DB)
        logger.logging.info("MY SQL connect successfully" )
        df=pd.read_sql_query("SELECT * from medical_insurance", connection_sql)
        logger.logging.info(f"Sample data {df.head}")
        return df
    
    except Exception as e:
        logger.logging.info(customeException(e, sys))
        raise customeException(e, sys)

