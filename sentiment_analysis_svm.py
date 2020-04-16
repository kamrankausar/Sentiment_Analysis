#!/usr/bin/env python
# coding: utf-8

# In[1]:


from platform import python_version
print(python_version())


# In[2]:


from sklearn.feature_extraction.text import TfidfVectorizer

import time
from sklearn import svm
from sklearn.metrics import classification_report

import pandas as pd


# In[3]:


# train Data
trainData = pd.read_csv("https://raw.githubusercontent.com/kamrankausar/Sentiment_Analysis/master/data/train.csv")


# In[4]:


trainData.columns


# In[5]:


trainData.head()


# In[6]:


# test Data
testData = pd.read_csv("https://raw.githubusercontent.com/kamrankausar/Sentiment_Analysis/master/data/test.csv")


# In[7]:


# Create feature vectors
vectorizer = TfidfVectorizer(min_df=5,      # Ignore terms that have a document frequency strictly lower than the given threshold
                             max_df = 0.8,  # Ignore terms that have a document frequency strictly higher than the given threshold
                             sublinear_tf=True, 
                             use_idf=True)  # Enable inverse-document-frequency reweighting


# In[8]:


train_vectors = vectorizer.fit_transform(trainData['Content'])


# In[9]:


test_vectors = vectorizer.transform(testData['Content'])


# In[10]:


# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()


# In[11]:


classifier_linear.fit(train_vectors, trainData['Label'])
t1 = time.time()


# In[12]:


prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()


# In[13]:


time_linear_train = t1-t0
time_linear_predict = t2-t1


# In[14]:


# results
print("Results for SVC(kernel=linear)")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))


# In[15]:


report = classification_report(testData['Label'], prediction_linear, output_dict=True)


# In[16]:


print('positive: ', report['pos'])
print('negative: ', report['neg'])


# In[17]:


# Test the Data
review = """SUPERB, I AM IN LOVE IN THIS PHONE"""


# In[18]:


review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))


# In[19]:


review = """Do not purchase this product. My cell phone blast when I switched the charger"""


# In[20]:


review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))


# # Saving the Vector and Model

# In[21]:


import pickle


# In[22]:


# pickling the vectorizer
pickle.dump(vectorizer, open('vectorizer.sav', 'wb'))


# In[23]:


# pickling the model
pickle.dump(classifier_linear, open('classifier.sav', 'wb'))


# # Load the Model and Predict

# In[24]:


vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
classifier = pickle.load(open('classifier.sav', 'rb'))


# In[25]:


text = 'Do not purchase this product. My cell phone blast when I switched the charger'


# In[26]:


text_vector = vectorizer.transform([text])
result = classifier.predict(text_vector)


# In[27]:


result


# In[ ]:





# In[ ]:





# In[ ]:




