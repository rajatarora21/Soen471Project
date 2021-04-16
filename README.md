# Soen471Project

# Abstract

For this project, we have decided to use a data analysis and classification on HR Analytics dataset to successfully determine whether or not a data scientist is looking for job change. The purpose of the analysis is to efficiently implement a screening process for potential employees. Inorder to screen thousands of applications, the native process can be time consuming and costly. Moreover,companies would know if a candidate is interested in the job. The required dataset contains insightful information to help classify the candidates according to their experience, education, demographics and so on.

# Introduction

The HR analytics dataset found on kaggle contains insightful information to understand the factors that may lead to a data scientist searching for a new job. Our goal is to properly classify data scientists as either looking for a job change or not given the finite set of features. It is also important to obtain significant classification metrics through the process of analysis, cleansing and transforming the required dataset. By analysing the dataset we have determined that it is imbalanced. This typically refers to a classification problem where the number of features of each class is not properly distributed. This may result in improper classification due to the severely skewed class distribution. Another problem we may come across would be noisy data. Noisy data is when the data contains an additional amount of meaningless information. The occurrence of noisy data may impact the classification of meaningful information and can negatively affect the accuracy of each prediction along with taking up an unnecessary amount of storage space . As previously mentioned the data preprocessing is a crucial step in this project, that allows us to obtain significant classification scores. Since this dataset comes with a higher rating, there are many similar studies that have been applied to it. Being a common classification problem you can see that these studies are using similar classification techniques. Albeit, the differences are applied in their approach to pre process data. The algorithms we found that are suitable for this project would be decision tree with hyperparameter tuning and random forest classification. It will be interesting to compare and find out the difference between their outcomes.

# Materials and Methods

The data analysis and classification on HR Analytics dataset requires multiple tools along with integration of external API’s. The data analysis will be carried forward with two machine learning algorithms. Initially, Python3 will be used as the platform to  perform the analysis. Along with internal Python3 libraries, machine learning APIs will be embedded such as numpy, pandas, scikit-learn and matplotlib. The required dataset on which the analysis will be performed will be available from Kaggle. This dataset is already split into their appropriate test and training sets. Initially before starting the analysis, data cleaning will be performed for example, looking for the missing variables, dropping the unnecessary columns and analyzing the relation between columns. Integration of Numpy will help to perform actions on arrays, as default arrays are relatively slower to process. Numpy is around 50x faster than default arrays. Pandas will be used for data manipulation and analysis. It offers data structures and operations for manipulating numerical tables and time series. Scikit-learn will be the major library used to apply algorithms on the dataset. It will provide various features like classification and regression. The HR data analysis will be performed on two classification algorithms , Decision tree and Random forest. Decision tree is a supervised machine learning algorithm that can be used for both classification and regression problems. The HR analytics decision tree will be used to predict whether the candidate is looking for a new job by giving the percentage based on the dataset. Decision tree is quite easy to implement. Moreover for producing effective results, Random forests algorithm will be used. Random Forest is a tree-based machine learning algorithm that leverages the power of multiple decision trees for making decisions. Each node in the decision tree works on a random subset of features to calculate the output. The random forest then combines the output of individual decision trees to generate the final output. Finally, for the visualization of data and results MatplotLib library will be used. MatplotLib provides an object oriented API for embedding plots like bar graphs, pie charts etc. MatplotLib is usable like Matlab and has the ability to use it in Python. Hence all the tools, methods and algorithms will help in performing HR data analytics.

# Results

Cleaning:
	
The HR analytics dataset provided from kaggle contains both a training and test set. The training set contains 19158 entries along with 14 columns, 13 of which are the features and lastly the target. One of the issues that arised was that this dataset contains a significant amount of missing data instances, 20733 in the training set and 2204 belonging to the test set. 

This is the distribution of missing unknown instances among each feature.
![Image of Missing info]
(https://github.com/rajatarora21/Soen471Project/blob/main/Graph:Plot/MissingInfo.png)

