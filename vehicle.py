#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[29]:


data=pd.read_excel('vehicle.xlsx')


# In[30]:


data.head()


# In[31]:


data.isnull().sum()


# In[32]:


data.dtypes


# In[33]:


data=data.drop(['Fuel_system','Bore','Engine_type','Compression_ratio','Stroke','City_mpg','Highway_mpg','No_of_doors'], axis=1)


# In[7]:


from sklearn.preprocessing import LabelEncoder


# In[8]:


labelEncoder = LabelEncoder()


# In[39]:


category_col=['Brand','Fuel_Type','Aspiration','Body_style','Drive_wheels','Engine_location','Cylinders']


# In[40]:


mapping_dict = {}
for col in category_col:
    data[col] = labelEncoder.fit_transform(data[col])

    le_name_mapping = dict(zip(labelEncoder.classes_,

                               labelEncoder.transform(labelEncoder.classes_)))

    mapping_dict[col] = le_name_mapping

print(mapping_dict)


# In[9]:


data.dtypes


# In[34]:


from sklearn.preprocessing import LabelEncoder


# In[35]:


data["Brand"] =LabelEncoder().fit_transform(data["Brand"])
data["Fuel_Type"] =LabelEncoder().fit_transform(data["Fuel_Type"])
data["Aspiration"] =LabelEncoder().fit_transform(data["Aspiration"])

data["Body_style"] =LabelEncoder().fit_transform(data["Body_style"])
data["Drive_wheels"] =LabelEncoder().fit_transform(data["Drive_wheels"])
data["Engine_location"] =LabelEncoder().fit_transform(data["Engine_location"])
#data["Engine_type"] =LabelEncoder().fit_transform(data["Engine_type"])
#data["Cylinders"] =LabelEncoder().fit_transform(data["Cylinders"])
#data["Fuel_system"] =LabelEncoder().fit_transform(data["Fuel_system"])


# In[15]:


#data=data.drop(['Wheel_base','Width','Length','Engine_type','Fuel_system','Stroke','City_mpg','Highway_mpg','No_of_doors'], axis=1,inplace=True)


# In[36]:


x = data.drop(['Risk'],axis=1)
y = data['Risk']


# In[37]:


x.dtypes


# In[18]:


# from sklearn.preprocessing import StandardScaler
# scaled_train = StandardScaler().fit_transform(X_train)
# scaled_test = StandardScaler().fit_transform(X_test)


# In[38]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=30)


# In[17]:


x_test.shape


# In[18]:


x_train.shape


# In[39]:


from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)


# In[44]:


from sklearn.tree import DecisionTreeClassifier
model= DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(x_train, y_train)
model.score(x_test,y_test)


# In[42]:


y_pred=model.predict(x_test)


# In[43]:


from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)


# In[48]:


# from sklearn.model_selection import cross_val_score
# cv=croos_validation_score()
# cross_val_score = (model, x_train, y_train, cv=10)


# In[49]:


import pickle
pickle.dump(model, open('model.pkl','wb'))


# In[ ]:




