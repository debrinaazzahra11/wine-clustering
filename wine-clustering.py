import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

df = pd.read_csv('wine-clustering.csv')

X = df.drop(['Nonflavanoid_Phenols', 'Alcohol', 'Malic_Acid', 'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols', 'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline'], axis=1)

st.header ("Dataset")
st.write(X)

# Menentukan Jumlah Cluster Dengan Elbow
clusters = []
for i in range(1, 11):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)

plt.figure(figsize=(12, 8))
sns.lineplot(x=list(range(1, 11)), y=clusters)
plt.title('Mencari Elbow')
plt.xlabel('Clusters')
plt.ylabel('Inertia')

plt.show()

st.set_option('deprecation.showPyplotGlobalUse', False)
elbo_plot = st.pyplot()

st.sidebar.subheader("Nilai jumlah K")
clust = st.sidebar.slider("Pilih Jumlah Cluster :", 2,10,3,1)

def k_means(n_clust):("Nilai jumlah K")
    kmean = KMeans(n_clusters=n_clust).fit(X)
    X['Labels'] = kmean.labels_
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X['Ash'], y=X['Magnesium'], hue=X['Labels'], palette=sns.color_palette('hls', n_clust))

    for label in X['Labels'].unique():
        plt.annotate(label,
            (X[X['Labels'] == label]['Ash'].mean(),
            X[X['Labels'] == label]['Magnesium'].mean()),
            horizontalalignment='center',
            verticalalignment='center',
            size=20, weight='bold',
            color='black')
        
    st.header('Cluster Plot')
    st.pyplot()
    st.write(X)
    
k_means(clust)
