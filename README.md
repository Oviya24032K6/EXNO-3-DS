## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/0df255b1-1f4d-4e31-84e8-08baf0a1a57e)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/6aecd1ee-96bb-4975-99a2-b09e36047f63)
```
 df['bo2']=e1.fit_transform(df[["ord_2"]])
 df
 ```

![image](https://github.com/user-attachments/assets/12f64615-0767-438a-82e0-3b6e6c8d914d)
```
 le=LabelEncoder()
 dfc=df.copy()
 dfc['ord_2']=le.fit_transform(dfc['ord_2'])
 dfc
```
![image](https://github.com/user-attachments/assets/6c5779a6-276a-4bb6-99db-1666f8e3622c)
```
 le=LabelEncoder()
 dfc=df.copy()
 dfc['ord_2']=le.fit_transform(dfc['ord_2'])
 dfc
```
![image](https://github.com/user-attachments/assets/613dc323-5760-48ef-a83f-af4481817d42)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/ea74aa6e-1a9d-4fbe-a1ec-9d36ba0e3432)
```
 pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/a1c0a852-ee11-4db6-9148-d2653960e852)
```
 pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/b5328f4b-ebfb-4f15-8a93-080518421a1f)
```
 from category_encoders import BinaryEncoder
 df=pd.read_csv("data.csv")
 df
```
![image](https://github.com/user-attachments/assets/dbb34920-0ba0-4fc3-bb97-c5f2eff5bcdd)
```
 be=BinaryEncoder()
 nd=be.fit_transform(df['Ord_2'])
 dfb=pd.concat([df,nd],axis=1)
 dfb
```
![image](https://github.com/user-attachments/assets/5b2825e7-4a0c-4c83-bf76-fe40cf207de1)
```
from category_encoders import TargetEncoder
 te=TargetEncoder()
 CC=df.copy()
 new=te.fit_transform(X=CC["City"],y=CC["Target"])
 CC=pd.concat([CC,new],axis=1)
 CC
```
![image](https://github.com/user-attachments/assets/8b1bf0d1-3e21-4caa-b1f0-c9453b75e194)
```
 import pandas as pd
 from scipy import stats
 import numpy as np
 df=pd.read_csv("Data_to_Transform.csv")
 df
```
![image](https://github.com/user-attachments/assets/efcd48d1-1079-420e-8ff6-159e33342f92)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/19202843-afee-4a90-90cd-a9ddf3ef4f7e)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/be7bf911-8f32-40f8-a514-0f59ad640fa2)
```
 np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/55de32ba-43a2-46d2-9b0b-31f96c098cd8)

  ```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/55cbd008-a50a-4a2f-9647-7ec296af735f)
```
 df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
 df
```
![image](https://github.com/user-attachments/assets/39980e46-9e79-4bf4-ad34-a230fc6ecfaf)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/29351901-60ce-4441-b426-84fc9b402fd6)
```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal')
 df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 df
```
![image](https://github.com/user-attachments/assets/95a1bd47-9586-41e6-a9fa-74afd507790a)
```
 import seaborn as sns
 import statsmodels.api as sm
 import matplotlib.pyplot as plt
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
```
![image](https://github.com/user-attachments/assets/b8b52929-7937-4c22-a435-bc5c0c9b8db0)
```
 sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
 plt.show()
```
![image](https://github.com/user-attachments/assets/6644aedc-0d73-4e33-8396-ed72f458e0db)
```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
```
![image](https://github.com/user-attachments/assets/0cf687d4-092b-4eb1-9bb1-91285295cd1a)



       
# RESULT:
     
       Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed.
