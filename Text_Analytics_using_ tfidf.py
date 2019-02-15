# -*- coding: utf-8 -*-
"""
SPAM SMS Project
We're going to classify spam versus non spam SMS.There's an attempt to make our own spam detector.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

df=pd.read_csv('SMSSpamCollection.csv',delimiter='\t', quoting=3)

df.head()

df.shape

df=pd.read_csv('SMSSpamCollection.csv',delimiter='\t', quoting=3,names=['Status','Message'])
df.head()

len(df[df.Status=='spam'])
len(df[df.Status=='ham'])
df['Status'].value_counts()
df.groupby('Status').describe()

#Dummy

df.loc[df["Status"]=='ham',"Status",]=0
df.loc[df["Status"]=='spam',"Status",]=1

df_x=df["Message"]
df_y=df["Status"]

# Vectorize

cv = TfidfVectorizer(min_df=1,stop_words='english')

#Splitting dataset

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

# vectorizing the model 
cv = TfidfVectorizer(min_df=1,stop_words='english')

# Fit to get vocab. and transform to get in matrix
x_traincv = cv.fit_transform(["Hi How are you How are you doing","Hi what's up","Wow that's awesome"])

# Convert into Dense matrix by array conversion
x_traincv.toarray()

cv.get_feature_names()

cv1 = TfidfVectorizer(min_df=1,stop_words='english')

x_traincv=cv1.fit_transform(x_train)

a=x_traincv.toarray()

a[0]

cv1.inverse_transform(a[0])

x_train.iloc[0]

# Testing

# No need for fit as we do not want this to learn vocab. of our data. We only transform
x_testcv=cv1.transform(x_test)

x_testcv.toarray()

# Model

mnb = MultinomialNB()

testmessage=x_test.iloc[0]

testmessage

predictions=mnb.predict(x_testcv[0])

predictions

a=np.array(y_test)

a

count=0

for i in range (len(predictions)):
    if predictions[i]==a[i]:
        count=count+1
        
len(predictions)
count
1068/1115.0

# Conclusion: Our model shows 95.7% accuracy