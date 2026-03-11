from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from preprocessing import preprocessing
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle





def models():
    kmeans = KMeans(4)
    pca = PCA(n_components=2)
    df = pd.read_csv('Customer Data.csv')
    df = df.drop(columns=['CUST_ID'])
    print(df.head())
    scaled_df = preprocessing()
    principal_components = pca.fit_transform(scaled_df)

    pca_df = pd.DataFrame(data = principal_components, columns=['PCA1','PCA2'])



    #Model Building using KMeans:

    kmeans.fit_predict(principal_components)

    pca_df_kmeans= pd.concat([pca_df,pd.DataFrame({'cluster':kmeans.labels_})],axis=1)


    # Creating a target column "Cluster" for storing the cluster segment:

    cluster_df = pd.concat([df,pd.DataFrame({'Cluster':kmeans.labels_})],axis=1)

    # Saving the kmeans clustering model and the data with cluster label:
    #Saving Scikitlearn models

    joblib.dump(kmeans, "kmeans_model.pkl")

    cluster_df.to_csv("Clustered_Customer_Data.csv")


    # Training and Testing the model accuracy using decision tree:

    x = cluster_df.drop(['Cluster'], axis=1)
    y = cluster_df[['Cluster']]

    x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.2)

    # Decision Tree:
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(x_train, y_train)

    prediction = model.predict(x_test)

    # Saving the Model :
    filename = 'final_model.sav'
    pickle.dump(model, open(filename, 'wb'))
 

    return pca_df_kmeans, cluster_df


if __name__ == "__main__":
    models()