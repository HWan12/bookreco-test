# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 19:09:18 2019

@author: Shreya
"""


from pyspark.ml.feature import StringIndexer





import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Utility:
    
    def __init__(self):
        print("Initializing Utlity class")
    
    def to_StringIndex(self,colname,dataframe):
        indexer = StringIndexer(inputCol=colname, outputCol=colname+'_Idx')
        dataframe = indexer.fit(dataframe).transform(dataframe)
        return dataframe
    
    
    
    
    