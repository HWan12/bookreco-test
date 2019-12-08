    # -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 20:55:35 2019

@author: Shreya
"""

from flask import Flask, request, render_template, session, redirect
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from utilityFile import Utility
from Recommend_App import RecommendationFacade
import pandas as pd
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import rand
import json


app = Flask(__name__)

appl = RecommendationFacade()

utility = Utility()

spark = SparkSession.builder.master('local').appName('recommender').getOrCreate()
#sqlContext = SQLContext(spark.sparkContext)

df_sampled = spark.read.json("Data/reviews_Books_5.json")
df_books = spark.read.json("Data/metaBooks.json")

df_sampled = df_sampled.select("reviewerID","asin","overall")
df_sampled = utility.to_StringIndex('reviewerID',df_sampled)
df_sampled = utility.to_StringIndex('asin',df_sampled)
#spark.sparkContext.
print("Splitting data for evaluation...")
(training_data, test_data) = df_sampled.randomSplit([0.8, 0.2],100)

engine = appl.return_engine_instance(spark,training_data,df_books)
    
als_model = appl.load_and_save_model(spark,training_data,engine)
als_model.save("Data/model")
#app.evaluate_model(spark,als_model,engine,test_data)

@app.route('/getbooks', methods=('GET','POST'))
def get_books(no_of_books):
    book_list=[]
    list_books = df_books.select('asin').orderBy(rand()).limit(no_of_books).collect()
    for book in list_books:
        book_list.append(book[0])
    json.dumps(book_list)
    
    
@app.route('/adduser', methods=('GET','POST'))    
def add_user():
    #als_model = ALS.load("C:\MBS-Rutgers programs\Big Data Algo\Project\data\model")
    df_newuser = appl.add_new_user(spark,als_model,df_books,10)
    new_user_predictions = appl.predict_ratings_new_user(spark,als_model,engine,df_newuser)
    new_user_predictions.createTempView("newuserreviews")
    df = spark.sql('select * from newuserreviews by prediction DESC').limit(10)
    df = df.toPandas()
    return render_template('simple.html',  tables=[df.to_html(classes='data', header="true")])

if __name__ == '__main__':
    app.run(debug=True)