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
Set the step=25 and bins=50, draw the graph for the numIterations. The accuracy is the highest when impurity is 'entropy'. But the AUCs are similar, only little difference.
~~~python
    impurity_list = ["gini", "entropy"]
    max_depth_list = [25]
    max_bins_list = [50]
~~~
![image](https://user-images.githubusercontent.com/75282285/194674555-9aabbbe1-edce-4206-b6d4-aa2b845d7981.png)

"maxDepth"
Set the impurity='entropy' and bins=50, draw the graph for the numIterations. The accuracy is the highest when depth=25. 
~~~python
    impurity_list = ["entropy"]
    max_depth_list = [3, 5, 10, 15, 20, 25]
    max_bins_list = [50]
~~~
![image](https://user-images.githubusercontent.com/75282285/194675429-5aa5ca31-26f9-4f14-825d-749caebf7bb8.png)


"maxBins"
Set the impurity='entropy' and depth=25, draw the graph for the numIterations. The accuracy is the highest when bins=50. 
~~~python
    impurity_list = ["entropy"]
    max_depth_list = [25]
    max_bins_list = [3, 5, 10, 50, 100, 200]
~~~
![image](https://user-images.githubusercontent.com/75282285/194675824-8ec50dc0-dee5-4760-ae58-fbb36fdc165e.png)


# Stage2: Train and evaluate   
Created the model using train data set.   
Calculated the accuracy using validation data set.
Sorted the metrics.    
Found the best parameters includ the best accuracy and the best model.  
~~~python
def train_evaluation_model(train_data,
                           validation_data,
                           impurity,
                           max_depth,
                           max_bins):
    start_time = time()
    model = DecisionTree.trainClassifier(train_data,
                                         numClasses=7,
                                         categoricalFeaturesInfo={},
                                         impurity=impurity,
                                         maxDepth=max_depth,
                                         maxBins=max_bins
                                         )
    accuracy = evaluate_model(model, validation_data)
    duration = time() - start_time
    return accuracy, impurity, max_depth, max_bins, duration, model
~~~
![image](https://user-images.githubusercontent.com/75282285/194676146-74caa6e2-2fac-4d4b-93b2-4328b2ab399f.png)


# Stage3: Test
Used the sub_test data set and the best model to calculate the AUC. If testing accuracy is similare as the best accuracy, it is OK.
As the result, the best accuracy is  0.8755, use the test data set to calcuate accuracyis 0.8571, the difference is 0.0006, so it has not overfitting issue. 
![image](https://user-images.githubusercontent.com/75282285/194676169-0910d5b3-d5dc-4fd2-9dae-80122dad488e.png)


# Stage4: Predict
~~~python
def predict_data(best_model):
    for lp in label_point_RDD.take(20):
        predict = best_model.predict(lp.features)
        label = lp.label
        features = lp.features
        result = ("Correct" if  (label == predict) else "Error")
        print("Forest：Elevation" + str(features[0]) +
                 " Aspect:" + str(features[1]) +
                 " Slope:" + str(features[2]) +
                 " Vertical_Distance_To_Hydrology :" + str(features[3]) +
                 " Horizontal_Distance_To_Hydrology:" + str(features[4]) +
                 " Hillshade_9am :" + str(features[5]) +
                 "....==>Predict:" + str(predict) +
                 " Fact:" + str(label) + "Result:" + result)
~~~
![image](https://user-images.githubusercontent.com/75282285/194675371-c2aa861c-9f4f-444b-9da4-1eccea269a02.png)


# Spark monitor

http://node1:8080/    
![image](https://user-images.githubusercontent.com/75282285/194676000-3acea0e2-9e02-40f2-8065-f143efef1eaa.png)
