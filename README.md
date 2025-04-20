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
  DEVELOPED BY : NARRA RAMYA
  REG NO : 212223040128
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/24d217e6-a34c-46a9-9739-538d9317398b)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
ramya=OrdinalEncoder(categories=[pm])
ramya.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/c2ff6b0b-8a58-4364-adf2-dc4495ea3eb7)
```
 df['bo2']=ramya.fit_transform(df[["ord_2"]])
 print(df)
```
![image](https://github.com/user-attachments/assets/ad52587e-9b8a-4606-a5b3-1bf770646ac1)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/4c0d4d16-c742-4504-a25e-8cf62e8bbf05)
```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
df2 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2 = pd.concat([df2, enc], axis=1)
df2
```
![image](https://github.com/user-attachments/assets/7575873a-368c-4d0a-8c39-0ed8098efed4)
```
 pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/14b8cce6-1775-4f75-81f8-663283474bba)
```
pip install category_encoders
```
![image](https://github.com/user-attachments/assets/370bb429-258f-4ec6-9297-495372823af1)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
![image](https://github.com/user-attachments/assets/92cc7d3c-f45a-4bfc-9a26-1956eb19dc90)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
![image](https://github.com/user-attachments/assets/5a7d9694-b22e-4770-99d8-06b0fee72076)
```
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/e0c5f140-393f-4b4f-b65e-d10affb4aefc)
```
 from category_encoders import TargetEncoder
 te=TargetEncoder()
 CC=df.copy()
 new=te.fit_transform(X=CC["City"],y=CC["Target"])
 CC=pd.concat([CC,new],axis=1)
 CC
```
![image](https://github.com/user-attachments/assets/4946955d-3a67-44c0-8749-99a70283d21e)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```

![image](https://github.com/user-attachments/assets/c31b9cdc-6b3c-49e9-aed1-0a3086da46d6)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/4b53c252-7daf-4fc0-9bf5-316e0a4c611c)
```
 np.log(df["Highly Positive Skew"])

```
![image](https://github.com/user-attachments/assets/b7f5caf0-b5bb-4b2b-ad79-d6d35b6d66ac)
```
 np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/8c33a2e6-89de-4608-88b1-e3aadb789126)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/72753577-bd18-48b2-b2eb-d0734a62fdae)
```
 np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/44882161-e35a-400e-9a2b-3d7a0c715123)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/eef53775-19e5-4cd2-80f2-778625d21760)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/70f0f3f7-7ba0-4189-b2b0-05e9d4dab9d0)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/6597db34-50b2-40fc-8316-f74af27bde51)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/47184ba5-921e-4fb2-af73-4d6834b871af)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/0e43805b-86d6-4444-adb0-9c3ce80285fc)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/fd423fe2-1236-48e4-bc1d-6f34fbe2937b)
```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show(
```
![image](https://github.com/user-attachments/assets/f64e418a-2e86-401a-b8ed-725269f9202b)
```
 df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
 sm.qqplot(df["Highly Negative Skew"],line='45')
 plt.show()
```
![image](https://github.com/user-attachments/assets/f269bbe6-02f2-42bb-8d01-b1b79a88f7c9)

# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
