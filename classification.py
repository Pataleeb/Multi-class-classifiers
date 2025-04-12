import scipy.io
import numpy as np
import random
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

data = scipy.io.loadmat('mnist_10digits.mat')
print(data.keys())

data['xtrain']=data['xtrain'].astype(float)/255.0
data['xtest']=data['xtest'].astype(float)/255.0
scaler = StandardScaler()
xtrain=scaler.fit_transform(data['xtrain'])
xtest=scaler.transform(data['xtest'])
data['ytrain']=data['ytrain'].ravel()
data['ytest']=data['ytest'].ravel()
##Training model
mlp=MLPClassifier(hidden_layer_sizes=(20,10),max_iter=500)
mlp.fit(data['xtrain'],data['ytrain'])
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(10, 10), learning_rate='constant',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)

##performance evaluation
predictions = mlp.predict(data['xtest'])
print(classification_report(data['ytest'], predictions,digits=4))
print(confusion_matrix(data['ytest'], predictions))

###Downsample
np.random.seed(6740)
downsample = np.random.choice(len(data['xtrain']),5000,replace=False)
xtrain_new=data['xtrain'][downsample]
ytrain_new=data['ytrain'][downsample]

###Best k for KNN
k_values=list(range(1,10))
cv_values=[]
for k in k_values:
       knn=KNeighborsClassifier(n_neighbors=k)
       scores=cross_val_score(knn,xtrain_new,ytrain_new,cv=5,scoring='accuracy')
       mean_val=scores.mean()
       cv_values.append(mean_val)
bestk =k_values[np.argmin(cv_values)]
print(f"Best k for KNN: {bestk}")

###Different classifiers
svm_kernel=SVC(kernel='rbf')
param_grid ={
       'C':[0.1,1,10,100],
       'gamma':[0.01,0.1,1,10]
}
grid_search = GridSearchCV(svm_kernel,param_grid,cv=5)
grid_search.fit(xtrain_new,ytrain_new)
print(f"Best SVM parameters: {grid_search.best_params_}")
###Best SVM
###Classifiers with optimal parameters
knn=KNeighborsClassifier(n_neighbors=2)
logistic_Reg=LogisticRegression(max_iter=2000,solver='lbfgs',multi_class='multinomial')
svm_kernel=SVC(kernel='rbf',C=10, gamma=0.01)
svm=SVC(kernel='linear',C=1, gamma='scale')

Classifiers= {
       "KNN" : knn,
       "LogisticRegression" : logistic_Reg,
       "SVM_kernel" : svm_kernel,
       "SVM" : svm}

for name, clf in Classifiers.items():
       print(f"\n Training {name}...")
       if name == "KNN" or name=="SVM":
              clf.fit(xtrain_new, ytrain_new)
       else:
              clf.fit(data['xtrain'], data['ytrain'])

       predictions=clf.predict(data['xtest'])

       accuracy=accuracy_score(data['ytest'], predictions)
       conf_matrix = confusion_matrix(data['ytest'], predictions)

       class_report=classification_report(data['ytest'], predictions,digits=4,output_dict=True)

       print(f"\n{name} Accuracy: {accuracy:.4f}")

       print("\nConfusion Matrix:")
       print(conf_matrix)

       print("\nPrecision, Recall, and F1-score:")
       for digit in sorted(class_report.keys()):
              if digit.isdigit():
                     print(f"Digit {digit}: Precision={class_report[digit]['precision']:.4f}, "
                           f"Recall={class_report[digit]['recall']:.4f}, "
                           f"F1-score={class_report[digit]['f1-score']:.4f}")