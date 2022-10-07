Data is pubic. We want to use ML to judge the forest's cover types (7 types) by multi factors. 

# Python_Spark_ML_DecisionTreeMulti    


Use DecisionTree and BinaryClassificationMetrics to find evergreen webside. 

Running environment is Spark + Hadoop + PySpark    
Used the algorithm is DecisionTree.     
Used the library is pyspark.mllib and MulticlassMetrics. 

# Stage1:  Read data
Placed the tsv on hadoop. Built 3 data sets: (1) Train data, (2) Validation data, (3) Sub_test data.

## Compare the parameters
"impurity"
~~~
    impurity_list = ["gini", "entropy"]
    max_depth_list = [15]
    max_bins_list = [10]
~~~
![image](https://user-images.githubusercontent.com/75282285/194674555-9aabbbe1-edce-4206-b6d4-aa2b845d7981.png)

