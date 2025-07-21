import pandas as pd
import numpy as np

import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap  


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier , NeighborhoodComponentsAnalysis,LocalOutlierFactor
from sklearn.decomposition import PCA


#WARNİNG LİBRARY
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("/Users/edanurunal/Desktop/data.csv")


df.drop(['Unnamed: 32' , 'id'], inplace = True, axis =1 )

df= df.rename(columns  =  {"diagnosis" : "target" })

sns.countplot(df["target"])
print(df.target.value_counts())

df["target"] = [1 if i.strip( ) == "M" else 0 for i  in df.target]

print(len(df))
print(df.head())
print(" Df shape" , df.shape)

df.info()

describe = df.describe()



#missing value none


#%%EDA

#correlation

# Spyder Console'a (aşağıya) yaz:
#%matplotlib qt5


corr_matrix= df.corr()
sns.clustermap(corr_matrix , annot= True, fmt= ".2f")
plt.title ("Correlation Between Features")
plt.show()

threshold = 0.5
filtre = np.abs(corr_matrix["target"]) > threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(df[corr_features].corr() , annot= True, fmt= ".2f")
plt.title("Correlation Between Features w Corr Threshold 0 .75")


# there some correlated features


#%%box plot


df_melted = pd.melt(df, id_vars = "target",  #verinin değişmeyecek kısmı o satırın hangi sınıfa ait olduğunu (0 mı 1 mi) 
                      var_name ="features",
                      value_name ="value" )




plt.figure()
sns.boxplot( x = "features", y ="value", hue = "target" , data = df_melted)
plt.xticks(rotation = 90)
plt.show()



#%%pair plot
sns.pairplot(df[corr_features],diag_kind = "kde", markers = "+",hue = "target")
plt.show()




#%%outlier
y = df.target
x = df.drop(["target"], axis= 1)
columns = x.columns.tolist()


# google sklearn.neighbours.LocalOutlierFactor

clf = LocalOutlierFactor()
clf.fit_predict(x)
X_score = clf.negative_outlier_factor_

outlier_score = pd.DataFrame()
outlier_score["score"] = X_score


#%%threshold
threshold = -2.5
filtre = outlier_score["score"] < threshold
outlier_index  =outlier_score [filtre].index.tolist()


plt.figure()
plt.scatter(x.iloc[outlier_index, 0], x.iloc[outlier_index, 1], color="blue", s=50, label="Outlier")
plt.scatter(x.iloc[:, 0], x.iloc[:, 1], color="k", s=3, label="Data points")




radius = (X_score.max() - X_score) / (X_score.max() - X_score.min())
outlier_score["radius"] = radius

plt.scatter(x.iloc[: , 0],x.iloc[: , 1], s =10000*radius , edgecolors= "r" , facecolors = "none", label = " Outlier Scores")


plt.legend()
plt.show()




#%%drop outliers

x = x.drop(outlier_index)
y = y.drop(outlier_index).values




#%%tarin-test split

test_size = 0.3

X_train, X_test, Y_train, Y_test = train_test_split(x,y , test_size = test_size, random_state = 42)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train_df = pd.DataFrame(X_train , columns = columns)


X_train_df_describe = X_train_df.describe()

X_train_df["target"] = Y_train



#%%box plot


df_melted = pd.melt(X_train_df, id_vars = "target",  #verinin değişmeyecek kısmı o satırın hangi sınıfa ait olduğunu (0 mı 1 mi) 
                      var_name ="features",
                      value_name ="value" )




plt.figure()
sns.boxplot( x = "features", y ="value", hue = "target" , data = df_melted)
plt.xticks(rotation = 90)
plt.show()



#%%pair plot
sns.pairplot(X_train_df[corr_features],diag_kind = "kde", markers = "+",hue = "target")
plt.show()







## %%KNN MWTHOD

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,Y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(Y_test, y_pred)
acc = accuracy_score(Y_test, y_pred)
score = knn.score(X_test , Y_test)
print("Score: " ,score)
print("CM:" ,cm)
print("Basic KNN Acc:" ,acc)

#%% choose best parameters


def KNN_BEST_PARAMS(x_train , x_test, y_train, y_test):
    
    k_range = list(range(1,31))
    weight_options = ["uniform","distance"]
    print()
    param_grid = dict(n_neighbors =  k_range, weights = weight_options)
    
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv = 10, scoring="accuracy")
    grid.fit(x_train, y_train)


    print("Best trainig score: {} with parameters: {}" .format(grid.best_score_, grid.best_params_))
    print()
   
    knn = KNeighborsClassifier(**grid.best_params_)
    knn.fit(x_train, y_train)
    
   
    y_pred_test = knn.predict(x_test)
    y_pred_train = knn.predict(x_train)
   
   
    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train , y_pred_train)

    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    print("Test Score: {}, Train Score:{} ".format(acc_test, acc_train))
    print()
    print("CM Test: " ,cm_test)
    print("CM Train: " ,cm_train)
   
    return grid

grid= KNN_BEST_PARAMS(X_train, X_test, Y_train, Y_test)


 #%%pca
 

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components= 2)
pca.fit(x_scaled)


X_reduced_pca =pca.transform(x_scaled)
pca_data = pd.DataFrame(X_reduced_pca,columns = ["p1","p2"])
pca_data["target"] = y
sns.scatterplot(x = "p1", y= "p2",hue="target" , data= pca_data )
plt.title("PCA: p1 vs p2")



X_train_pca, X_test_pca, Y_train_pca, Y_test_pca = train_test_split(X_reduced_pca,y , test_size = test_size, random_state = 42)



grid_pca = KNN_BEST_PARAMS(X_train_pca , X_test_pca, Y_train_pca, Y_test_pca)




#visualise
cmap_light = ListedColormap(['orange', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'darkblue'])

# step size in the mesh
h = .05 
X = X_reduced_pca
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = grid_pca.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("i-Class classification (k = %i, weights = '%s')"
          % (len(np.unique(y)), grid_pca.best_estimator_.n_neighbors,grid_pca.best_estimator_.weights ))




#%% NCA

nca = NeighborhoodComponentsAnalysis(n_components = 2 , random_state=42)
nca.fit(x_scaled , y)
X_reduced_nca =nca.transform(x_scaled)
nca_data = pd.DataFrame(X_reduced_nca, columns = ["p1", "p2"])
nca_data["target"] = y
sns.scatterplot(x="p1", y="p2", hue="target", data =nca_data)
plt.title("Nca : p1 vs p2")




X_train_nca, X_test_nca, Y_train_nca, Y_test_nca = train_test_split(X_reduced_nca,y , test_size = test_size, random_state = 42)



grid_nca = KNN_BEST_PARAMS(X_train_nca , X_test_nca, Y_train_nca, Y_test_nca)


#1 tane hata için görselleştirme

#visualise
cmap_light = ListedColormap(['orange', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'darkblue'])

# step size in the mesh
h = .2 #kası
X = X_reduced_nca
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = grid_nca.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("i-Class classification (k = %i, weights = '%s')"
          % (len(np.unique(y)), grid_nca.best_estimator_.n_neighbors,grid_nca.best_estimator_.weights ))




# %%find wrong decision
knn = KNeighborsClassifier(**grid_nca.best_params_)
knn.fit(X_train_nca, Y_train_nca)
y_pred_nca = knn.predict(X_test_nca)
acc_test_nca = accuracy_score(y_pred_nca, Y_test_nca)
knn.score(X_test_nca, Y_test_nca)

test_data = pd.DataFrame()
test_data["x_test_nca_p1"] = X_test_nca[:,0]
test_data["x_test_nca_p2"] = X_test_nca[:,1]
test_data["y_pred_nca"] = y_pred_nca
test_data["Y_test_nca"] = Y_test_nca

plt.figure()
sns.scatterplot(x="x_test_nca_p1", y="x_test_nca_p2", hue="y_test_nca", data=test_data)

diff = np.where(y_pred_nca != Y_test_nca)[0]
plt.scatter(test_data.iloc[diff,0], test_data.iloc[diff,1], label = "Wrong Classified")





















