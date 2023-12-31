import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

df = pd.read_csv('wine-clustering.csv')

X = df.drop(['Nonflavanoid_Phenols', 'Alcohol', 'Malic_Acid', 'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols', 'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline'], axis=1)

st.header ("Dataset Wine Cluster")
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

st.sidebar.subheader("Nilai jumlah Wine Cluster")
clust = st.sidebar.slider("Pilih Jumlah Wine Cluster :", 2,7,3,1)

def k_means(n_clust):
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
        
    st.header('Wine Cluster Plot')
    st.pyplot()
    st.write(X)

    st.header('Wine Cluster Plot')
    st.write(X)

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with colors based on Labels
    scatter = ax.scatter(
    df['Ash'],
    df['Ash_Alcanity'],
    df['Magnesium'],
    c=kmeans.labels_,
    cmap='viridis',
    s=50  # Marker size
    )

    # Customize the plot
    ax.set_xlabel('Ash')
    ax.set_ylabel('Ash_Alcanity')
    ax.set_zlabel('Magnesium')
    ax.set_title('KMeans Clustering in 3D')

    # Add a colorbar
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
    colorbar.set_label('Labels')

    
k_means(clust)
