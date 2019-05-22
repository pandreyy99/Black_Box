import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import  XGBClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
# from sklearn import preprocessing
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pdb
from sklearn.model_selection import GridSearchCV

from data_loader import *

data_loader = Data_Loader()
data_loader.build_dataset()
train_data, test_data = data_loader.train_data, data_loader.test_data
print("Si-ncepe...")

########################################
#       Scaling before applying PCA    #
########################################

def normalize_data(train_data, test_data, type='none'):
    # ({‘none’,‘standard’ , ‘min_max’ , ‘l1’ , ‘l2’ })

    scaler = None

    if type == 'standard':
        scaler = preprocessing.StandardScaler()

    elif type == 'min_max':
        # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # (0, 1) default
        scaler = preprocessing.MinMaxScaler()
    elif type == 'l1':
        scaler = preprocessing.Normalizer('l1')
    elif type == 'l2':
        # scaler = preprocessing.Normalizer() implicit l2
        scaler = preprocessing.Normalizer('l2')

    if scaler is not None:
        scaler.fit(train_data)
        scaled_x_train = scaler.transform(train_data)
        scaled_x_test = scaler.transform(test_data)
        return scaled_x_train, scaled_x_test
        print("scaled")
    else:
        return train_data, test_data

# Here I apply Standard scaler on data

train_datas , test_datas = normalize_data(train_data, test_data, 'standard')
scaler = preprocessing.StandardScaler()
scaler.fit(train_data)
eval_data = scaler.transform(data_loader.eval)

# Principal Component Analysis
pca = PCA(.85)
pca.fit(train_data)
train_datas = pca.transform(train_data)
test_datas = pca.transform(test_data)
eval_data = pca.transform(data_loader.eval)

# print('Train data shape : ', np.shape(train_datas))
# print('Test data shape : ', np.shape(test_datas))
# print('eval data shape : ', np.shape(eval_data))

# print("test")
# print(len(data_loader.test_data))
# print("train")
# print(len(data_loader.train_data))
# print(len(data_loader.train_labels))
# print(data_loader.num_classes)

def get_accuracy(eticheta1, eticheta2):
    return len(eticheta1[eticheta1 == eticheta2]) / len(eticheta1)

def get_accuracy_statistics(train_data, test_data, Cs, normalization_type='none', svm_type='linear'):
    # 1 Aceasta normalizează datele
    train_data, test_data = normalize_data(train_data, test_data, normalization_type)

    # 2 Antrenează câte un SVM pentru fiecare valoare din C

    clase_train = np.zeros(len(Cs))
    clase_test = np.zeros(len(Cs))
    for index in range(len(Cs)):
        svm_classifier = svm.SVC(C=Cs[index], kernel=svm_type)
        svm_classifier.fit(train_data, data_loader.train_labels)
        clase_train[index] = get_accuracy(svm_classifier.predict(train_data), data_loader.train_labels)
        clase_test[index] = get_accuracy(svm_classifier.predict(test_data), data_loader.test_labels)

    # 3 Returnează 2 vectori conținând acuratețea fiecărui model pe datele de antrenare, respectiv de test

    return clase_train, clase_test

'''
def build_confusion(etichete_reale, etichete_prezize):
    clase = max(etichete_reale) + 1
    confuzie = np.zeros((clase,clase))
    for i in range(clase):
        for j in range(clase):
            perechi = [(et1,et2) for (et1, et2) in zip(etichete_reale, etichete_prezize) if et1 == i and et2 == j]
            confuzie[i,j] = len(perechi)
    return confuzie

def precision(confusion_matrix, length):
    precisionArray = np.zeros(length)
    for i in range(length):
        sum = 0
        for j in range(length):
            sum += confusion_matrix[i,j]
        precisionArray[i] = confusion_matrix[i,i] / sum
    return  precisionArray
'''
# # Cs = [1e-8, 1e-7, 1e-6, 1]
# Cs = [1e-7, 1e-6, 1e-4, 1e-2, 1e-1, 0.5, 1, 10, 100]
# Cs = [1e-4, 1e-2, 1e-1, 0.2e-1, 0.5, 1, 10, 100 ]
# Cs = [100]

# train_data = np.reshape(train_data, (len(train_data),4096))
# test_data = np.reshape(test_data, (len(test_data),4096))

###################################################
#                       SVM
###################################################

#file_to_write_scores = open('scores.txt', 'a')

# std_accuracies_train, std_accuracies_test = get_accuracy_statistics(train_datas, test_datas, Cs, 'standard')
# file_to_write_scores.write('std accuracies train : ' +  str(std_accuracies_train) + '\n')
# file_to_write_scores.write('std acc test : ' + str(std_accuracies_test) + '\n')
# std_accuracies_train, std_accuracies_test = get_accuracy_statistics(train_datas, test_datas, Cs, 'standard', 'rbf')
# print(std_accuracies_train, end='\n')
# print(std_accuracies_test + '\n', end='\n')

# classifier = svm.SVC()
# print("It's fitting")
# classifier.fit(train_datas, data_loader.train_labels)
# score = classifier.score(test_datas, data_loader.test_labels)
#
# print("score : " , end='\n')
# print(score,end='\n')


normalize_data(train_datas, test_datas, 'standard')
normalize_data(train_datas, eval_data, 'standard')
'''
rf_classifier = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

CV_rfc = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv= 5)
CV_rfc.fit(train_data, data_loader.train_labels)
print(CV_rfc.best_params_)
best_p1 = CV_rfc.best_params_
'''
rfc = RandomForestClassifier(criterion='entropy', max_depth=8, max_features='auto', n_estimators=100, random_state=42)
rfc.fit(train_data, data_loader.train_labels)
predicted_labels = rfc.predict(data_loader.eval)
# score = rfc.score(test_data, data_loader.test_labels)
# print("score : " + str(score), end='\n')

# from sklearn.model_selection import cross_val_score

# scores = cross_val_score(rfc, train_data, data_loader.train_labels, cv=3)
# scores = np.array(scores)
# mean_score = scores.mean()
# print("Mean Score : " + str(mean_score))

from sklearn.model_selection import KFold

kf = KFold(n_splits=3)

file_to_write_confusion_matrix = open('confusion_matrix.txt', 'w')

for train_index, test_index in kf.split(train_data):
    X_train, X_test = train_data[train_index], train_data[test_index]
    y_train, y_test = data_loader.train_labels[train_index], data_loader.train_labels[test_index]
    rfc.fit(X_train, y_train)
    confusion_matrixx = confusion_matrix(y_test, rfc.predict(X_test))
    np.savetxt('confusion_matrix.txt',confusion_matrixx)
    # file_to_write_confusion_matrix.write()

'''
CV_rfc = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv= 5)
CV_rfc.fit(train_datas, data_loader.train_labels)
print(CV_rfc.best_params_)
best_p2 = CV_rfc.best_params_
'''


# file_to_write_scores.write('std rbf accuracies train : ' +  str(std_accuracies_train) + '\n')
# file_to_write_scores.write('std rbf acc test : ' + str(std_accuracies_test) + '\n')
# std_accuracies_train, std_accuracies_test = get_accuracy_statistics(train_datas, test_datas, Cs, 'standard', 'sigmoid')
# file_to_write_scores.write('std sigmoid accuracies train : ' +  str(std_accuracies_train) + '\n')
# file_to_write_scores.write('std sigmoid acc test : ' + str(std_accuracies_test) + '\n')

# print("de aici printeaza",end='\n')

'''
perceptron = Perceptron( penalty='l2', tol=1e-5, shuffle=True, eta0= 0.5, n_jobs= -1, early_stopping=True, n_iter_no_change= 10)
perceptron.fit(train_datas, data_loader.train_labels)
score = perceptron.score(test_datas, data_loader.test_labels)
print(score,end='\n')
# file_to_write_scores.write('Perceptron : ' + str(score) + '\n')

perceptron = Perceptron( penalty='l1', tol=1e-7, shuffle=True, eta0= 0.2, n_jobs= -1, early_stopping=True, n_iter_no_change= 10)
perceptron.fit(train_datas, data_loader.train_labels)
score = perceptron.score(test_datas, data_loader.test_labels)
print(score,end='\n')

perceptron = Perceptron( penalty='elasticnet', tol=1e-5, shuffle=True, eta0= 0.5, n_jobs= -1, early_stopping=True, n_iter_no_change= 10)
perceptron.fit(train_datas, data_loader.train_labels)
score = perceptron.score(test_datas, data_loader.test_labels)
print(score,end='\n')
'''
'''
file_to_write_scores.write('Here starts mlps : ' + '\n')
mlp = MLPClassifier(activation='tanh', solver='lbfgs', learning_rate='adaptive', shuffle=True, verbose=True, early_stopping=True)
mlp.fit(train_datas, data_loader.train_labels)
score = mlp.score(test_datas, data_loader.test_labels)
print(score,end='\n')
file_to_write_scores.write(str(score) + '\n')

mlp = MLPClassifier(activation='logistic', solver='lbfgs', learning_rate='adaptive', shuffle=True, verbose=True, early_stopping=True)
mlp.fit(train_datas, data_loader.train_labels)
score = mlp.score(test_datas, data_loader.test_labels)
print(score,end='\n')
file_to_write_scores.write(str(score) + '\n')

mlp = MLPClassifier(activation='relu', solver='lbfgs', learning_rate='adaptive', shuffle=True, verbose=True, early_stopping=True)
mlp.fit(train_datas, data_loader.train_labels)
score = mlp.score(test_datas, data_loader.test_labels)
print(score,end='\n')
file_to_write_scores.write(str(score) + '\n')

mlp = MLPClassifier(activation='tanh', solver='sgd', learning_rate='adaptive', shuffle=True, verbose=True, early_stopping=True)
mlp.fit(train_datas, data_loader.train_labels)
score = mlp.score(test_datas, data_loader.test_labels)
print(score,end='\n')
file_to_write_scores.write(str(score) + '\n')
'''
#############################################################
#    BEST       BEST        BEST        BEST        BEST
#############################################################

'''
mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate='adaptive', shuffle=True, early_stopping=True)
mlp.fit(train_datas, data_loader.train_labels)
score = mlp.score(test_datas, data_loader.test_labels)
print(score,end='\n')
file_to_write_scores.write(str(score) + '\n')
'''

'''
mlp = MLPClassifier(activation='relu', solver='sgd', learning_rate='adaptive', shuffle=True, verbose=True, early_stopping=True)
mlp.fit(train_datas, data_loader.train_labels)
score = mlp.score(test_datas, data_loader.test_labels)
print(score,end='\n')
file_to_write_scores.write(str(score) + '\n')

mlp = MLPClassifier(activation='tanh', solver='adam', learning_rate='adaptive', shuffle=True, verbose=True, early_stopping=True)
mlp.fit(train_datas, data_loader.train_labels)
score = mlp.score(test_datas, data_loader.test_labels)
print(score,end='\n')
file_to_write_scores.write(str(score) + '\n')

mlp = MLPClassifier(activation='logistic', solver='adam', learning_rate='adaptive', shuffle=True, verbose=True, early_stopping=True)
mlp.fit(train_datas, data_loader.train_labels)
score = mlp.score(test_datas, data_loader.test_labels)
print(score,end='\n')
file_to_write_scores.write(str(score) + '\n')

mlp = MLPClassifier(activation='relu', solver='adam', learning_rate='adaptive', shuffle=True, verbose=True, early_stopping=True)
mlp.fit(train_datas, data_loader.train_labels)
score = mlp.score(test_datas, data_loader.test_labels)
print(score,end='\n')
file_to_write_scores.write(str(score) + '\n')

# train_data = np.reshape(train_data, (len(train_data),4096))
# test_data = np.reshape(test_data, (len(test_data),4096))

####################################################
#             Kernel Ridge Regresion(KRR)
####################################################

krr_classifier = KernelRidge(kernel='rbf')
krr_classifier.fit(train_datas, data_loader.train_labels)
score = krr_classifier.score(test_datas, data_loader.test_labels)
print(score,end='\n')
file_to_write_scores.write('KRR : ' + str(score) + '\n')

##########################
#       KNN
###########################

knn_clasifier = KNeighborsClassifier(weights='distance', algorithm='auto')
knn_clasifier.fit(train_datas, data_loader.train_labels)
score = knn_clasifier.score(test_datas, data_loader.test_labels)
print(score,end='\n')
file_to_write_scores.write('KNN : ' + str(score) + '\n')

# etichete = krr_classifier.predict(test_data)
#print(etichete)
'''
################################################
#           Submission
################################################
# print("Predict : ")
#
# predicted_labels = classifier.predict(data_loader.eval)
#
# import csv
#
# with open('sample_submission.csv', 'w', newline='') as csv_file:
#     writer = csv.writer(csv_file)
#     # header = ["Id" , "Prediction"]
#     writer.writerow(["Id", "Prediction"])
#     for i in range(len(predicted_labels)):
#         list = []
#         list.append(i + 1)
#         list.append(predicted_labels[i])
#         writer.writerow(list)

'''
predicted_labels = mlp.predict(eval_data)
predicted_labels = mlp.predict(data_loader.eval)
'''


for label in predicted_labels:
    print(label , end=" ")
print('\n\n\n')

import csv

with open('sample_submission-random_forest_classifier2.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # header = ["Id" , "Prediction"]
    writer.writerow(["Id", "Prediction"])
    for i in range(len(predicted_labels)):
        list = []
        list.append(i + 1)
        list.append(predicted_labels[i])
        writer.writerow(list)

'''
xgb1 = XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8,colsample_bytree=0.8,
                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)

param_grid = {
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2),
    'gamma':[i/10.0 for i in range(0,5)],
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)],
    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}

xgb_grid = GridSearchCV(xgb1, param_grid=param_grid, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
xgb_grid.fit(train_data, data_loader.train_labels)
print("xgb_grid score : " + str(xgb_grid.best_score_))
print("best param : ",end='\n')
print(xgb_grid.best_params_)
params = open('params.txt', 'w')
params.write("Best Score : ")
params.write(xgb_grid.best_score_)
params.write("Best Params : ")
params.write(xgb_grid.best_params_)

xgbc = XGBClassifier()
xgbc.fit(train_datas, data_loader.train_labels)
predicted_labels_xgbc = xgbc.predict(eval_data)
score = xgbc.score(test_datas, data_loader.test_labels)
print("XGB score : " + str(score), end='\n')

for label in predicted_labels_xgbc:
    print(label , end=" ")
print('\n\n\n')

import csv

with open('sample_submission-XGBC.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # header = ["Id" , "Prediction"]
    writer.writerow(["Id", "Prediction"])
    for i in range(len(predicted_labels_xgbc)):
        list = []
        list.append(i + 1)
        list.append(predicted_labels_xgbc[i])
        writer.writerow(list)
'''