# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 08:41:14 2019

@author: Shreya
"""
from pyspark.sql import SparkSession

from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType, DoubleType

from recommendation import RecommendationEngine
from utilityFile import Utility
import random
#import pyspark
from pyspark.sql.functions import rand

class RecommendationFacade:
    
    def __init__(self):
        print("Welcome to my engine")
        
        
    def return_engine_instance(self,spark,training_data,df_books):
        print("Returning a recommendation engine...")
        return RecommendationEngine(spark,training_data,df_books)
        
    """Create an ALS model and save for future prediction """   
    def load_and_save_model(self,spark,training_data,reco_engine):
        print("Creating and saving model...")
        als_model = reco_engine.train_model(training_data,20,10,"reviewerID_Idx","asin_Idx","overall")
        return als_model
     
    def evaluate_model(self,spark,als_model,reco_engine,test_data):
        print("Evaluating model")
        rmse = reco_engine.evaluate_model(als_model,test_data)
        print("RMSE of the model is: ",rmse)
        
      
          
    """Manually creating a new user and selecting some books for prediction """   
    def add_new_user(self,spark,als_model,df_books):
        print("Adding new user...")
        new_user_ratings=[]
        utility = Utility()
        ## Creating a random user Id
        new_user_id = ''.join(random.choice('0123456789ABCDEF') for i in range(14))
        
        ## Selecting random books from the books metadata
        list_books = df_books
        for book in list_books:
            record = (new_user_id,book)  #,float(random.randint(0,5))"""
            new_user_ratings.append(record) 
            #del record
        
        ## Creating a schema for the newuser dataframe.
        schema = StructType([
                 StructField("reviewerID", StringType(), True),
                 StructField("asin", StringType(), True),
                 #StructField("overall", DoubleType(), True)
                ])   
        ## Creating the dataframe for the new user including the reviewerID and the book ids
        df_newuser = spark.createDataFrame(new_user_ratings, schema)   
        ##Converting the reviewerId and the bookId's to numeric fields   
        df_newuser = utility.to_StringIndex('reviewerID',df_newuser)
        df_newuser = utility.to_StringIndex('asin',df_newuser)
        return df_newuser
    
    def predict_ratings_new_user(self,spark,als_model,reco_engine,df_newuser):
       # print("Predicting ratings for a new user")
        new_user_predictions = reco_engine.predict_ratings(als_model,df_newuser)
        
        #new_user_predictions.sort("prediction",ascending=False)
        return new_user_predictions


