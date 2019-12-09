#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 11:35:41 2019

@author: hwan
"""

import os
#from pyspark.sql import SparkSession
from flask import Flask, flash, redirect, render_template, request, url_for, session
import pandas as pd
#from utilityFile import Utility
#from Recommend_App import RecommendationFacade
#from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.functions import rand
import pyspark
#from pyspark.sql import SQLContext

VIZ_FOLDER = os.path.join('static', 'Viz')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = VIZ_FOLDER



#spark = SparkSession.builder.master('local').appName('recommender').getOrCreate()
'''
appl = RecommendationFacade()

utility = Utility()



df_rb = spark.read.json("Data/reviews_Books_5.json")
df_mb = spark.read.json("Data/metaBooks.json")

df_rb = df_rb.select("reviewerID","asin","overall")
df_rb = utility.to_StringIndex('reviewerID',df_rb)
df_rb = utility.to_StringIndex('asin',df_rb)

engine = appl.return_engine_instance(spark,df_rb,df_mb)
    
#als_model = appl.load_and_save_model(spark,df_rb,engine)
'''
df_mb = spark.read.json("gs://sh_books_bucket/metaBooks.json")
#df_rb = spark.read.json("gs://sh_books_bucket/reviews_Books_5.json")
#model = ALSModel.load('Data/model')

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/explore3')
def explore():
    viz1 = os.path.join(app.config['UPLOAD_FOLDER'], 'rating dist.png')
    viz2 = os.path.join(app.config['UPLOAD_FOLDER'], 'rating by time.png')
    viz3 = os.path.join(app.config['UPLOAD_FOLDER'], 'rating by sales rank.png')
    return render_template("explore3.html", viz1 = viz1, viz2=viz2, viz3=viz3)

@app.route('/recommendation')
def recommendation():
    book_list=[]
    list_books = df_mb.where(df_mb.title!='None').select(['asin','title']).orderBy(rand()).limit(10).collect()
    for book in list_books:
        book_list.append((book['asin'],book['title'])) 
    return render_template("recommendation.html",
                           book_id0 = book_list[0][0], book_title0=book_list[0][1],
                           book_id1 = book_list[1][0], book_title1=book_list[1][1],
                           book_id2 = book_list[2][0], book_title2=book_list[2][1],
                           book_id3 = book_list[3][0], book_title3=book_list[3][1],
                           book_id4 = book_list[4][0], book_title4=book_list[4][1],
                           book_id5 = book_list[5][0], book_title5=book_list[5][1],
                           book_id6 = book_list[6][0], book_title6=book_list[6][1],
                           book_id7 = book_list[7][0], book_title7=book_list[7][1],
                           book_id8 = book_list[8][0], book_title8=book_list[8][1],
                           book_id9 = book_list[9][0], book_title9=book_list[9][1],
                           )
  
@app.route('/result', methods=('GET','POST'))    
def result():
    select=[]
    for i in range(10):
        val = request.form.getlist('rating'+str(i))
        select.append(val)
    while ([] in select):
        select.remove([])
    book_rating=[val[0].split(' ') for val in select]
    df = pd.DataFrame(book_rating, 
               columns =['overall', 'asin'])

        
    #df_newuser = appl.add_new_user(spark,model,select)
    #new_user_predictions = appl.predict_ratings_new_user(spark,model,engine,df_newuser)
    #new_user_predictions.createTempView("newuserreviews")
    #df = spark.sql('select * from newuserreviews by prediction DESC')
    #df = df_newuser.toPandas()
    
    return render_template('result.html',  tables=[df.to_html(classes='data', header="true")])
    #return render_template('result.html', select=book_rating)


if __name__=='__main__':
    app.run(debug=True)
