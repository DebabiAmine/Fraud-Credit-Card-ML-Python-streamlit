import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("application de Machine Learning pour la detection de Fraude par carte de credit ")
    st.subheader("Auteur : Mohamed Amine Debabi")
    #fonction d'importation des données
    @st.cache(persist=True) #enregistre les données sans reexcuter la data
    def load_data():
        data = pd.read_csv('creditcard.csv')
        return data
    # Affichage la table de données
    df = load_data()
    df_sample = df.sample(100)

    X = df.drop('Class', axis = 1)
    if st.sidebar.checkbox("Afficher lesDonnees brutes", False):
        st.subheader("jeu de données 'creditcard' : Echantillon de 100 observations ")
        st.write(df_sample)

    if st.sidebar.checkbox("Shape", False):
        st.subheader("SHAPE ")
        st.write(df.shape)


    if st.sidebar.checkbox("describe", False):
        st.subheader("describe ")
        st.write(df.describe())

    if st.sidebar.checkbox("SUM Null", False):
        st.subheader("SUM Null ")
        st.write(df.isnull().sum()/df.shape[0]*100)

    seed = 123

    #train/test Split

    y = df['Class']
    X = df.drop('Class', axis = 1)
    X_train, X_test, y_train, y_test=train_test_split(
        X, df['Class'],
            test_size=0.2,
            stratify=df['Class'],
            random_state=seed
        )



    class_names=['T.Authentique','T.Frauduleuse']


    classifier = st.sidebar.selectbox(
      "Classificateur",
        ("Random Forest","SVM", "Logistic Regression")
    )

    #Ananlyse de la performance des modeles
    def plot_pref(graphes):
        if 'Confusion matrix' in graphes:
            st.subheader('Matrice de confusion')
            ConfusionMatrixDisplay.from_estimator(
              model,
                X_test,
                y_test,
                display_labels=class_names
            )
            st.pyplot()

        if 'ROC Curve' in graphes:
            st.subheader('Courbe ROC')
            RocCurveDisplay.from_estimator(
              model,
            X_test,
            y_test,

            )
            st.pyplot()
        if 'Precision-Recall curve' in graphes:
            st.subheader('Courbe Precision-Recall curve')
            PrecisionRecallDisplay.from_estimator(
              model,
                X_test,
                y_test,

            )
            st.pyplot()


    #Random Forest
    if classifier == "Random Forest":
        st.sidebar.subheader("Hyperparametres du modele")
        n_arbres = st.sidebar.number_input(
            "choisir le nombre d'arbres dans la foret",
            100, 1000, step=10
        )
        profondeur_arbre = st.sidebar.number_input(
            "Choisir la Profondeur maximale d'un arbre",
            1,20,step=1
        )
        bootstrapp = st.sidebar.radio(
            "Echantillons Bootstrap lors de la creation d'arbres ?",
            (True,False)
        )
        graphes_pref = st.sidebar.multiselect(
            "Choisir un graphique de performance du modele ML",
            ("Confusion matrix", "ROC Curve", "Precision-Recall curve")
        )

        if st.sidebar.button("Excution",key="classify"):
            st.subheader("Random Forest Results")
            #Initialisation d'un objet RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=n_arbres,
                max_depth=profondeur_arbre,
                bootstrap=bootstrapp,
                random_state=seed
            )
            #Entrainement de l'algorithme
            model.fit(X_train, y_train)

            #Prediction
            y_pred = model.predict(X_test)

            #Metrique de performance
            accuracy = model.score(X_test, y_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            #Afficher les metriques dans l'application

            st.write("Accuracy:",accuracy.round(3))
            st.write("Precision:", precision.round(3))
            st.write("Recall:", recall.round(3))

            #Afficher les graphiques de performance
            plot_pref(graphes_pref)

    # Regression logistique
    if classifier == "Logistic Regression":
            st.sidebar.subheader("Hyperparametres du modele")
            hyp_c = st.sidebar.number_input(
                "choisir la valeur du parametre de régularisation",
                0.01, 10.0
            )
            n_max_iter = st.sidebar.number_input(
                "Choisir le nombre maximum d'iterations",
                100, 1000, step=10
            )

            graphes_pref = st.sidebar.multiselect(
                "Choisir un graphique de performance du modele ML",
                ("Confusion matrix", "ROC Curve", "Precision-Recall curve")
            )

            if st.sidebar.button("Excution", key="classify"):
                st.subheader("Logistic Regression Results")
                # Initialisation d'un objet Logistic Regression
                model = LogisticRegression(
                    C=hyp_c,
                    max_iter=n_max_iter,
                    random_state=seed
                )
                # Entrainement de l'algorithme
                model.fit(X_train, y_train)

                # Prediction
                y_pred = model.predict(X_test)

                # Metrique de performance
                accuracy = model.score(X_test, y_test)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                # Afficher les metriques dans l'application

                st.write("Accuracy:", accuracy.round(3))
                st.write("Precision:", precision.round(3))
                st.write("Recall:", recall.round(3))

                # Afficher les graphiques de performance
                plot_pref(graphes_pref)

    # SVM
    if classifier == "SVM":
        st.sidebar.subheader("Hyperparametres du modele")
        hyp_c = st.sidebar.number_input(
            "choisir la valeur du parametre du SVM",
            0.01, 10.0
        )

        kernel = st.sidebar.radio(
            "Choisir le Kernel",
            ("rbf", "linear")
        )

        gamma = st.sidebar.radio(
            "Gamma",
            ("scale","auto")
        )

        graphes_pref = st.sidebar.multiselect(
            "Choisir un graphique de performance du modele ML",
            ("Confusion matrix", "ROC Curve", "Precision-Recall curve")
        )

        if st.sidebar.button("Excution", key="classify"):
            st.subheader("Support Vector Machine (SVM) Results")
            # Initialisation d'un objet SVc
            model = SVC(
                C=hyp_c,
                kernel=kernel,
                gamma=gamma
            )
            # Entrainement de l'algorithme
            model.fit(X_train, y_train)

            # Prediction
            y_pred = model.predict(X_test)

            # Metrique de performance
            accuracy = model.score(X_test, y_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Afficher les metriques dans l'application

            st.write("Accuracy:", accuracy.round(3))
            st.write("Precision:", precision.round(3))
            st.write("Recall:", recall.round(3))

            # Afficher les graphiques de performance
            plot_pref(graphes_pref)





if __name__ == '__main__':
    main()
