from pyspark import SparkConf, SparkContext
import numpy as np
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import MulticlassMetrics
from time import time
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.mllib.tree import DecisionTree


def create_spark_context():
    global sc, path
    sc = SparkContext(conf=SparkConf().setAppName('RunDecisionTreeBinary'))
    path = "hdfs://node1:8020/input/"


def read_data():
    global lines
    raw_data = sc.textFile(path + "covtype.data")
    lines = raw_data.map(lambda x: x.split(','))


def convert_float(x):
    result = 0 if x == "?" else float(x)
    return result


def extract_features(record, feature_end):
    result = [convert_float(i) for i in record[0:feature_end]]
    return result


def extract_label(field):
    label = field[-1]
    result = float(label)-1
    return result


def prepare_data():
    global label_point_RDD
    print("Before standard:")
    label_RDD = lines.map(lambda x: extract_label(x))
    feature_RDD = lines.map(lambda r: extract_features(r, len(r)-1))
    # for i in feature_RDD.first():
    #     print(f"{i},")
    print("After standard:")
    label_point = label_RDD.zip(feature_RDD)
    label_point_RDD = label_point.map(lambda x: LabeledPoint(x[0], x[1]))
    result = label_point_RDD.randomSplit([8, 1, 1])
    return result


def evaluate_model(model, validation_data):
    score = model.predict(validation_data.map(lambda x: x.features))
    score_and_labels = score.zip(validation_data.map(lambda x: x.label)).map(lambda x: (float(x[0]), float(x[1])))
    metrics = MulticlassMetrics(score_and_labels)
    accuracy = metrics.accuracy
    return accuracy


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


def eval_parameter(train_data, validation_data):
    impurity_list = ["gini", "entropy"]
    max_depth_list = [3, 5, 10, 15]
    max_bins_list = [3, 5, 10, 50]
    my_metrics = [
        train_evaluation_model(train_data,
                               validation_data,
                               impurity,
                               max_depth,
                               max_bins)
        for impurity in impurity_list
        for max_depth in max_depth_list
        for max_bins in max_bins_list
    ]
    s_metrics = sorted(my_metrics, key=lambda x: x[0], reverse=True)
    best_parameter = s_metrics[0]
    print(best_parameter)
    print(f"the best inpurity is:{best_parameter[1]}\n"
          f"the best max_depth is:{best_parameter[2]}\n"
          f"the best max_bins is:{best_parameter[3]}\n"
          f"the best accuracy is:{best_parameter[0]}\n")
    best_accuracy = best_parameter[0]
    best_model = best_parameter[5]
    return best_accuracy, best_model


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


def show_chart(df, eval_parm, bar_parm, line_parm, y_min=0.5, y_max=1.0):
    ax = df[bar_parm].plot(kind='bar', title=eval_parm, figsize=(10, 6), legend=True, fontsize=12)
    ax.set_xlabel(eval_parm, fontsize=12)
    ax.set_ylim([y_min, y_max])
    ax.set_ylabel(bar_parm, fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(df[line_parm].values, linestyle='-', marker='o', linewidth=2, color='r')
    plt.show()


def draw_graph(train_data, validation_data, draw_type):
    impurity_list = ["gini", "entropy"]
    max_depth_list = [3, 5, 10, 15, 20, 25]
    max_bins_list = [3, 5, 10, 50, 100, 200]
    if draw_type == "impurity":
        my_index = impurity_list
        max_depth_list = [25]
        max_bins_list = [50]
    elif draw_type == "maxDepth":
        my_index = max_depth_list
        impurity_list = ["entropy"]
        max_bins_list = [50]
    elif draw_type == "maxBins":
        my_index = max_bins_list
        impurity_list = ["entropy"]
        max_depth_list = [25]
    my_metrics = [
        train_evaluation_model(train_data,
                               validation_data,
                               impurity,
                               max_depth,
                               max_bins)
        for impurity in impurity_list
        for max_depth in max_depth_list
        for max_bins in max_bins_list
    ]
    df = pd.DataFrame(my_metrics,
                      index=my_index,
                      columns=['accuracy', 'duration', 'impurity', 'maxDepth', 'maxBins', 'model'])
    show_chart(df, draw_type, 'accuracy', 'duration', 0.5, 1.0)


if __name__ == "__main__":
    s_time = time()
    create_spark_context()
    print("Reading data stage".center(60, "="))
    read_data()
    train_d, validation_d, test_d = prepare_data()
    print(train_d.first())
    train_d.persist()
    validation_d.persist()
    test_d.persist()
    print("Draw".center(60, "="))
    draw_graph(train_d, validation_d, "impurity")
    draw_graph(train_d, validation_d, "maxDepth")
    draw_graph(train_d, validation_d, "maxBins")
    print("Evaluate parameter".center(60, "="))
    b_accuracy, b_model = eval_parameter(train_d, validation_d)
    print(f"The best accuracy is: {b_accuracy}")
    print("Test".center(60, "="))
    test_data_auc = evaluate_model(b_model, test_d)
    print(f"best auc is:{format(b_accuracy, '.4f')}, test_data_auc is: {format(test_data_auc, '.4f')}, "
          f"they are only slightly different:{format(abs(float(b_accuracy) - float(test_data_auc)), '.4f')}")
    print("Predict".center(60, "="))
    predict_data(b_model)

    train_d.unpersist()
    validation_d.unpersist()
    test_d.unpersist()









