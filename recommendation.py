# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:50:28 2019

@author: Shreya
"""
import pyspark
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from time import time
import logging

from utilityFile import Utility

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

## Removing the spark instances wherever required

"""A Book recommendation engine"""
class RecommendationEngine:
    
    
    """Init the recommendation engine given a Spark session and a dataset path"""
    def __init__(self,spark,df_sampled,df_books):
        
        logger.info("Starting up the Recommendation Engine: ")
        
        self.utility = Utility()
        self.spark = spark
        # Load ratings data for later use
        self.df_sampled = df_sampled
        # Load books data for later use
        self.data_books = df_books
        #self.spark.read.json("C:\MBS-Rutgers programs\Big Data Algo\Project\data\metaBooks.json")
        #self.train_model("reviewerID_Idx","asin_Idx","overall") 
 
    
    """Train the ALS model with the current dataset"""
    def train_model(self,training_data,rank,iterations,userCol,itemCol,ratingCol):
    
        logger.info("Training the ALS model...")

        #for rank in ranks:
        als = ALS(userCol=userCol, itemCol=itemCol, ratingCol=ratingCol,rank=rank,
                         maxIter=iterations, regParam=.05, nonnegative=True,coldStartStrategy="drop", implicitPrefs=False)
        
        """Pre-processing the alphanumeric columns to numeric value"""
        
        
        t0 = time()
        als_model = als.fit(training_data)
        tt = time() - t0
        
        logger.info("New model trained in %s seconds",round(tt,3))
        
        #als_model.save("gs://sh_books_bucket/code/model")
        return als_model
        
        
        
    def evaluate_model(self,als_model,test_data_df):
        
        logger.info("Evaluating Model...")
        
        data_to_predict = test_data_df.drop("overall")
        
        test_data = als_model.transform(data_to_predict)
        test_predictions = test_data_df.join(test_data,on=['reviewerId','asin'],how='inner').select(test_data_df.reviewerID,test_data_df.asin,test_data_df.overall,test_data.prediction)
        
        t0 = time()
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="overall",predictionCol="prediction")
        tt = time() - t0
        
        logger.info("Time taken to predict in %s seconds",round(tt,3))
        rmse = evaluator.evaluate(test_predictions)
             
        logger.info("RMSE:",rmse)
        return rmse
    
       
        
    """Gets predictions for a new (reviewer, asin) """
    
    def predict_ratings(self,als_model,new_user_to_predict):
    
        t0 = time()
        new_user_predictions = als_model.transform(new_user_to_predict)
        tt = time() - t0
        print ("New model evaluation in %s seconds",round(tt,3))
        #new_user_predictions = new_user_predictions.join(new_user_to_predict,on='asin',how='inner')
        new_user_predictions = new_user_predictions.join(self.data_books,on='asin',how='inner')
        new_user_predictions = new_user_predictions.select(new_user_predictions.reviewerID,new_user_predictions.asin,self.data_books.title,new_user_predictions.prediction)
               
        return new_user_predictions
    
    
    """Recommends up to books_count top unrated books to user_id """
    def get_ratings_unrated_books(self,als_model,user_id):
    
        # Get pairs of (userID, movieID) for user_id unrated books
        user_unrated_books = self.df_sampled.filter(~ (self.df_sampled.reviewerID == user_id)).select('asin')
        user_unrated_books = user_unrated_books.withColumn('reviewerID',user_id)
        user_unrated_books = self.utility.to_StringIndex('reviewerID',user_unrated_books)
        user_unrated_books = self.utility.to_StringIndex('asin',user_unrated_books)
        
        # Get predicted ratings
        ratings = self.predict_ratings(als_model,user_unrated_books)

        return ratings
    


    
    
    
    

