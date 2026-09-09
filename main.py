from src.ML_PROJECT import logger
from src.ML_PROJECT.exception import customeException
import sys
from src.ML_PROJECT.components.data_ingestion import DataIngestion

try:
    dataIngestionobj=DataIngestion()
    dataIngestionobj.read_data()
    
except Exception as e:
    logger.logging.info(customeException(e, sys))
    raise Exception(e, sys)