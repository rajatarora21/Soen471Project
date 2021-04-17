# Soen471Project

# Abstract

For this project, we have decided to use a data analysis and classification on HR Analytics dataset to successfully determine whether or not a data scientist is looking for job change. The purpose of the analysis is to efficiently implement a screening process for potential employees. Inorder to screen thousands of applications, the native process can be time consuming and costly. Moreover,companies would know if a candidate is interested in the job. The required dataset contains insightful information to help classify the candidates according to their experience, education, demographics and so on.

# Introduction

The HR analytics dataset found on kaggle contains insightful information to understand the factors that may lead to a data scientist searching for a new job. Our goal is to properly classify data scientists as either looking for a job change or not given the finite set of features. It is also important to obtain significant classification metrics through the process of analysis, cleansing and transforming the required dataset. By analysing the dataset we have determined that it is imbalanced. This typically refers to a classification problem where the number of features of each class is not properly distributed. This may result in improper classification due to the severely skewed class distribution. Another problem we may come across would be noisy data. Noisy data is when the data contains an additional amount of meaningless information. The occurrence of noisy data may impact the classification of meaningful information and can negatively affect the accuracy of each prediction along with taking up an unnecessary amount of storage space . As previously mentioned the data preprocessing is a crucial step in this project, that allows us to obtain significant classification scores. Since this dataset comes with a higher rating, there are many similar studies that have been applied to it. Being a common classification problem you can see that these studies are using similar classification techniques. Albeit, the differences are applied in their approach to pre process data. The algorithms we found that are suitable for this project would be decision tree with hyperparameter tuning and random forest classification. It will be interesting to compare and find out the difference between their outcomes.

# Materials and Methods

The data analysis and classification on HR Analytics dataset requires multiple tools along with integration of external API’s. The data analysis will be carried forward with two machine learning algorithms. Initially, Python3 will be used as the platform to  perform the analysis. Along with internal Python3 libraries, machine learning APIs will be embedded such as numpy, pandas, scikit-learn and matplotlib. The required dataset on which the analysis will be performed will be available from Kaggle. This dataset is already split into their appropriate test and training sets. Initially before starting the analysis, data cleaning will be performed for example, looking for the missing variables, dropping the unnecessary columns and analyzing the relation between columns. Integration of Numpy will help to perform actions on arrays, as default arrays are relatively slower to process. Numpy is around 50x faster than default arrays. Pandas will be used for data manipulation and analysis. It offers data structures and operations for manipulating numerical tables and time series. Scikit-learn will be the major library used to apply algorithms on the dataset. It will provide various features like classification and regression. The HR data analysis will be performed on two classification algorithms , Decision tree and Random forest. Decision tree is a supervised machine learning algorithm that can be used for both classification and regression problems. The HR analytics decision tree will be used to predict whether the candidate is looking for a new job by giving the percentage based on the dataset. Decision tree is quite easy to implement. Moreover for producing effective results, Random forests algorithm will be used. Random Forest is a tree-based machine learning algorithm that leverages the power of multiple decision trees for making decisions. Each node in the decision tree works on a random subset of features to calculate the output. The random forest then combines the output of individual decision trees to generate the final output. Finally, for the visualization of data and results MatplotLib library will be used. MatplotLib provides an object oriented API for embedding plots like bar graphs, pie charts etc. MatplotLib is usable like Matlab and has the ability to use it in Python. Hence all the tools, methods and algorithms will help in performing HR data analytics.

# Results

### Cleaning:
	
The HR analytics dataset provided from kaggle contains both a training and test set. The training set contains 19158 entries along with 14 columns, 13 of which are the features and lastly the target. After analyzing the data we noticed that it contains a significant amount of missing data instances, 20733 in the training set and 2204 belonging to the test set.

This is the distribution of missing unknown instances among each feature.

![MissingInfo](https://user-images.githubusercontent.com/47232584/115093868-a9e72b00-9ee9-11eb-9397-b3fe84d7b55f.png)

In order to prevent biased estimates we decided to drop columns company_size and company_type from the features list since the percentage of missing values in these columns is greater than 30%.

![Screen Shot 2021-04-16 at 7 49 20 PM](https://user-images.githubusercontent.com/47232584/115094810-e1a3a200-9eec-11eb-8dde-2a1daa6b91ff.png)



### Data Preprocessing:

To handle the remaining missing values the technique we decided to use is called imputing with most frequent values. Essentially, what it does is replaces the missing value with the mean or mode within that column. This approach also works with categorical data as it will just assign missing values the most frequent category that occurs in the specific column.

![Screen Shot 2021-04-16 at 8 56 08 PM](https://user-images.githubusercontent.com/47232584/115097105-40214e00-9ef6-11eb-8891-75651605dcf9.png)

### SMOTE

Another issue we came across is that the target class is imbalanced. This can be analyzed with the following graph where 0 - are data scientists not looking for a job change and 1 - are data scientists looking for job change.

![Screen Shot 2021-04-16 at 9 02 43 PM](https://user-images.githubusercontent.com/47232584/115097273-22081d80-9ef7-11eb-83bd-c81d35b0f5ca.png)

A problem with imbalanced classification is that there are too few examples of the minority class for a model to effectively learn the decision boundary. Smote is the techique used in our project to rectify this problem. It is a technique used to oversample the target values in the minority class. This is achieved by selecting a minory class instance at random, then finds its k-nearest neighbour and with these two values its able to create a new similar instance. The downfall to this algorithm is that the target values in the majority class are overlooked. Here is the class target after upsampling.

![Screen Shot 2021-04-16 at 9 02 50 PM](https://user-images.githubusercontent.com/47232584/115097393-f46fa400-9ef7-11eb-803b-35177fb7f49e.png)

### Classification algorithms

The classification models used in this project are Decision Tree's with and without Hyperparameter tuning and Random Forest. Pythons function GridSearchCV was used to find the most optimal Decision tree hyperparamters. GridSearch works by accepting a list of user defined parameters, then it tries all the combination with each value passed and then returns the ones that achieve the best performance. Here we can see the difference in performance between both Decision Tree models.

<img width="479" alt="Screen Shot 2021-04-16 at 9 24 55 PM" src="https://user-images.githubusercontent.com/47232584/115097808-5204f000-9efa-11eb-9d8a-0be969418523.png">
<img width="496" alt="Screen Shot 2021-04-16 at 9 25 07 PM" src="https://user-images.githubusercontent.com/47232584/115097820-5f21df00-9efa-11eb-93ca-37635e72db9c.png">

The decision tree with optimal parameters was able to increase the amount of correct classifications. Therefore resulting in better evaluation metrics.

Lastly, our random forest classifier achieved the best results. Random forest classifiers are known to achieve a better accuracy score with respect to other classification models. It works by taking an average of prediction outcomes from n number of decision trees. Here are the metrics of our model.

![Screen Shot 2021-04-16 at 9 54 09 PM](https://user-images.githubusercontent.com/47232584/115098488-529f8580-9efe-11eb-969c-32e7d1eb3b96.png)





