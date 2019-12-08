#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 19:23:57 2019

@author: hwan
"""

import pyspark
from pyspark.sql import SQLContext

import pyspark.sql.functions as F
from pyspark.sql.types import DateType
from pyspark.ml.feature import Bucketizer
import pandas as pd
import matplotlib.pyplot as plt


sc = pyspark.SparkContext('local[1]')
sqlContext = SQLContext(sc)

df_rb = sqlContext.read.json('Data/reviews_Books_5.json')
df_mb = sqlContext.read.json('Data/metaBooks.json')

#df_rb_ct = df_rb.crosstab('reviewerID','asin')
def flatten_df(nested_df):
    flat_cols = [c[0] for c in nested_df.dtypes if c[1][:6] != 'struct']
    nested_cols = [c[0] for c in nested_df.dtypes if c[1][:6] == 'struct']

    flat_df = nested_df.select(flat_cols +
                               [F.col(nc+'.'+c).alias(nc+'_'+c)
                                for nc in nested_cols
                                for c in nested_df.select(nc+'.*').columns])
    return flat_df


df_mb_flat = flatten_df(df_mb)


df_mb_short = df_mb_flat.select('asin','description','imUrl','price','title','salesRank_Books')

df_rb = df_rb.withColumn('reviewTime',F.from_unixtime('unixReviewTime').cast(DateType()))

df_rb_short = df_rb.drop('helpful').drop('reviewerName').drop('unixReviewTime')

df_join = df_rb_short.join(df_mb_short, df_rb_short.asin == df_mb_short.asin,'left').drop(df_mb_short.asin)
cols_todrop=['reviewText','summary','description','imUrl','title']
df_join_short = df_join.drop(*cols_todrop)
df_join_short = df_join_short.withColumn('reviewYear',F.year('reviewTime'))
max_rank = df_join_short.agg({'salesRank_Books':'max'}).collect()[0]["max(salesRank_Books)"]
bucketizer = Bucketizer(splits=[i * 5000 for i in list(range(int(max_rank/5000)+2))],
                                inputCol="salesRank_Books", outputCol="salesRank_Books_buckets")
df_join_short = bucketizer.setHandleInvalid("keep").transform(df_join_short)
df_join_short = df_join_short.withColumn('salesRank_ranged',
                                         df_join_short.salesRank_Books_buckets*5000)
df_join_short = df_join_short.drop('salesRank_Books_buckets')

#Visualizations
#viz1
viz1_df = df_join_short.groupBy('overall').count().toPandas() #distribution of ratings
plt.bar(viz1_df['overall'],viz1_df['count'])
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Rating Distribution')
plt.show()
#viz2
viz2_df = df_join_short.groupBy('reviewYear').agg(F.avg('overall'),F.count('overall')).orderBy('reviewYear').toPandas()
fig, ax1 = plt.subplots()
ax1.bar(viz2_df['reviewYear'], viz2_df['count(overall)'], color='b', alpha=0.5,tick_label=viz2_df['reviewYear'])
plt.xticks(rotation='vertical')
ax1.set_xlabel('Review Year')
ax1.set_ylabel('Rating Count', color='b')
[tl.set_color('b') for tl in ax1.get_yticklabels()]
ax2 = ax1.twinx()
ax2.plot(viz2_df['reviewYear'], viz2_df['avg(overall)'], 'r-')
ax2.set_ylabel('Average Rating', color='r')
[tl.set_color('r') for tl in ax2.get_yticklabels()]
plt.title('Rating by Time')
plt.show()
#viz3
viz3_df = df_join_short.groupBy('salesRank_ranged').agg(F.avg('overall'),F.count('overall')).orderBy('salesRank_ranged').toPandas()
plt.plot(viz3_df['salesRank_ranged'],viz3_df['avg(overall)'])
plt.xlabel('Sales Rank')
plt.ylabel('Average Rating')
plt.title('Rating by Sales Rank')
plt.show()

'''
fig, ax1 = plt.subplots()
ax1.bar(viz3_df['salesRank_ranged'], viz3_df['count(overall)'], color='b', alpha=0.5)
ax1.set_xlabel('Sales Rank')
ax1.set_ylabel('Rating Count', color='b')
[tl.set_color('b') for tl in ax1.get_yticklabels()]
ax2 = ax1.twinx()
ax2.plot(viz3_df['salesRank_ranged'], viz3_df['avg(overall)'], 'r-')
ax2.set_ylabel('Average Rating', color='r')
[tl.set_color('r') for tl in ax2.get_yticklabels()]
plt.title('Rating by Sales Rank')
plt.show()
'''