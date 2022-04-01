import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

colnames = ['surgery', 'age', 'hospital number', 'rectal temperature', 'pulse', 'respiratory rate', 'temperature of extremities', 'peripheral pulse', 'mucous membranes', 'capillary refill time', 'pain', 'peristalsis', 'abdominal distension', 'nasogastric tube',
            'nasogastric reflux', 'nasogastric reflux PH', 'rectal examination', 'abdomen', 'packed cell volume', 'total protein', 'abdominocentesis appearance', 'abdomcentesis total protein', 'outcome', 'surgical lesion', 'lesion type', 'lesion type2', 'lesion type3', 'pathology data']
dropped_colnames = ['nasogastric reflux PH', 'abdominocentesis appearance', 'abdomcentesis total protein',
                    'abdomen', 'nasogastric tube', 'nasogastric reflux', 'rectal examination', 'lesion type2', 'lesion type3']
train_colnames = set(colnames).difference(dropped_colnames)
train_colnames = [x for x in train_colnames if x != "outcome"]
cat_colnames = ['surgery', 'age', 'temperature of extremities', 'peripheral pulse', 'mucous membranes', 'capillary refill time', 'pain', 'peristalsis', 'abdominal distension',
                'surgical lesion', 'pathology data']
numerical_colnames = ['rectal temperature', 'pulse', 'respiratory rate',
                      'packed cell volume', 'total protein']

# Read dataset file
def read_data(dataset_name):
    loc = "data\%s" % (dataset_name)
    df = pd.read_csv(loc,
                     sep=' ', names=colnames, header=None)
    df.columns.names = ['id']
    df.drop(dropped_colnames, axis=1, inplace=True)
    df = df.replace('?', np.nan)
    df = df.dropna(subset=['outcome'])
    # split into data and target
    X, y = df.loc[:, df.columns != 'outcome'], df.loc[:, 'outcome']
    return X, y


X, y = read_data('horse-colic.data')

# Cast Object types to float
for col in train_colnames:
    X = X.astype({col: float})

# Pre-processing
filled_df = X.copy()
# Replacing missing values of categorical attributes with the most frequant value
cat_fill = filled_df.loc[:, cat_colnames]
for col in cat_colnames:
    filled_df[cat_colnames] = cat_fill.fillna(filled_df[col].mode()[0])

# Replacing missing values of numerical attributes with the mean value
filled_df[numerical_colnames] = filled_df[numerical_colnames].fillna(
    filled_df[numerical_colnames].mean())

# Normalization
normalized_df = filled_df.copy()
to_norm = normalized_df.loc[:, numerical_colnames]
normalized_df[numerical_colnames] = preprocessing.normalize(to_norm)

# split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(normalized_df, y, test_size=0.2)

# K-means
refined_df = normalized_df.copy()

kmeans = KMeans(algorithm='auto', copy_x=True, init='random', max_iter=300,
                n_clusters=3, n_init=10,
                random_state=0, tol=0.01, verbose=0)
k_clusters = kmeans.fit_predict(refined_df)
k_clusters += 1
refined_df['clusters'] = pd.Series(k_clusters)

n = 0
fig = plt.figure(figsize=(16, 28))
for col in train_colnames:
    p = plt.subplot(7, 4, n + 1)
    p.set_title(train_colnames[n])
    plt.scatter(refined_df[col], refined_df['clusters'],
                c=refined_df['clusters'], cmap='rainbow')
    n += 1

# Hieratical
refined_df2 = normalized_df.copy()

# Draw Dendrogram
plt.figure(figsize=(16, 16))
plt.title("Horse Colic Dendograms")
dend = shc.dendrogram(shc.linkage(
    refined_df2, method='complete'), truncate_mode='lastp')
plt.ylim(0, 65000)

Aglo = AgglomerativeClustering(
    n_clusters=3, affinity='euclidean', linkage='ward')
h_clusters = Aglo.fit_predict(refined_df2)
h_clusters += 1
refined_df2['clusters'] = pd.Series(h_clusters)

n = 0
plt.figure(figsize=(16, 28))
for col in train_colnames:
    p = plt.subplot(7, 4, n + 1)
    p.set_title(train_colnames[n])
    plt.scatter(refined_df2[col], refined_df2['clusters'],
                c=refined_df2['clusters'], cmap='rainbow')
    n += 1

# Classification Report
# y = y.to_numpy().astype(int)
print("K-meansHieratical clustering algorithm evaluation metrics:\n\n",
      classification_report(y, k_clusters))
print("__________________________________________________")
print("Hieratical clustering algorithm evaluation metrics:\n\n",
      classification_report(y, h_clusters))
