# Importación de las bibliotecas necesarias
import streamlit as st
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Función principal de la aplicación
def main():

    st.title('Clasificación de datos con K-Nearest Neighbors')

    # Subida del archivo CSV
    file_upload = st.file_uploader('Sube el archivo csv', type=['csv'])

    if file_upload is not None:

        df = pd.read_csv(file_upload)
        st.write(df)

        # Opción de manejo de datos faltantes
        if df.isnull().values.any():
            st.warning('Tu dataset tiene valores faltantes. ¿Cómo te gustaría manejarlos?')
            missing_option = st.selectbox('Elige una opción', ['Eliminar filas con valores faltantes', 'Rellenar con la media', 'Rellenar con la mediana', 'Rellenar con la moda'])
            if missing_option == 'Eliminar filas con valores faltantes':
                df.dropna(inplace=True)
            elif missing_option == 'Rellenar con la media':
                df.fillna(df.mean(), inplace=True)
            elif missing_option == 'Rellenar con la mediana':
                df.fillna(df.median(), inplace=True)
            else:
                df.fillna(df.mode().iloc[0], inplace=True)

        # Opciones de preprocesamiento de datos
        preprocessing_option = st.selectbox('Elige una opción de preprocesamiento', ['Ninguno', 'Escala Standard (Z-Score)', 'Normalización Min-Max', 'PCA'])
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if preprocessing_option == 'Escala Standard (Z-Score)':
            scaler = StandardScaler()
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        elif preprocessing_option == 'Normalización Min-Max':
            scaler = MinMaxScaler()
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

        elif preprocessing_option == 'PCA':
            pca = PCA(n_components=3)
            principal_components = pca.fit_transform(df[numeric_columns])
            df_pca = pd.DataFrame(data = principal_components, columns = ['Principal Component 1', 'Principal Component 2', 'Principal Component 3'])
            df_non_numeric = df.select_dtypes(exclude=['float64', 'int64'])
            df = pd.concat([df_pca, df_non_numeric], axis = 1)
            scaler = StandardScaler()  # Añade esto

            
        # Visualización de los datos de entrada
        if st.checkbox('Mostrar gráficos de los datos de entrada'):
            for column in df.columns:
                if is_numeric_dtype(df[column]):
                    fig = px.histogram(df, x=column, color_discrete_sequence=['indianred'], title=f"Histograma de {column}")
                    fig.update_layout(autosize=False, width=600, height=400, margin=dict(l=50, r=50, b=100, t=100, pad=4),
                                      paper_bgcolor="white",)
                    st.plotly_chart(fig)
            if st.checkbox('Mostrar gráficos de dispersión para cada par de características'):
                fig = px.scatter_matrix(df)
                st.plotly_chart(fig)
            if st.checkbox('Mostrar gráficos de caja y bigotes para resumir las características'):
                for column in df.columns:
                    if is_numeric_dtype(df[column]):
                        fig = px.box(df, y=column)
                        st.plotly_chart(fig)
            if st.checkbox('Mostrar gráficos de correlación para mostrar las relaciones entre las características'):
                corr = df.corr()
                fig = px.imshow(corr, labels=dict(color="Coeficiente de Correlación"),
                                x=corr.columns, y=corr.columns, color_continuous_scale='viridis', title="Mapa de Calor de Correlación")
                st.plotly_chart(fig)

        # Selección de columnas
        columns = df.columns.tolist()
        selected_columns = st.multiselect('Elige las columnas para entrenamiento', columns)
        target_variable = st.selectbox('Elige la variable objetivo', columns)

        if selected_columns and target_variable:  # Comprobamos que se seleccionaron columnas y una variable objetivo
            # Crear conjuntos de datos de entrenamiento y prueba
            X = df[selected_columns]
            y = df[target_variable]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Opciones de hiperparámetros
            n_neighbors = st.slider('Selecciona el número de vecinos para KNN', 1, 10, 5)
            weights_option = st.selectbox('Selecciona el tipo de pesos', ['uniform', 'distance'])
            algorithm_option = st.selectbox('Selecciona el algoritmo para calcular los vecinos más cercanos', ['auto', 'ball_tree', 'kd_tree', 'brute'])

            knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights_option, algorithm=algorithm_option)
            knn.fit(X_train, y_train)

            st.write('Modelo de Clasificación KNN entrenado')

            # Validación cruzada
            if st.checkbox('Usar Validación Cruzada'):
                cv = st.slider('Selecciona el número de folds para la Validación Cruzada', 2, 10, 5)
                cv_scores = cross_val_score(knn, X, y, cv=cv)
                st.write(f'Puntuaciones de Validación Cruzada: {cv_scores}')
                st.write(f'Precisión de Validación Cruzada: {cv_scores.mean()}')

            # Predicción y evaluación del modelo
            y_pred = knn.predict(X_test)

            st.write('Reporte de clasificación')
            st.text(classification_report(y_test, y_pred))

            # Gráfico de la matriz de confusión
            if st.checkbox('Mostrar gráfico de la matriz de confusión'):
                conf_matrix = confusion_matrix(y_test, y_pred)
                fig = px.imshow(conf_matrix, labels=dict(x="Predicted", y="Actual", color="Density"),
                                x=y.unique(), y=y.unique(), color_continuous_scale='viridis', title="Matriz de Confusión")
                fig.update_layout(autosize=False, width=600, height=400, margin=dict(l=50, r=50, b=100, t=100, pad=4),
                                  paper_bgcolor="white",)
                st.plotly_chart(fig)


            # Gráfico 3D de los datos
            if st.checkbox('Mostrar gráfico 3D'):
                if len(selected_columns) > 2:
                    fig = px.scatter_3d(df, x=selected_columns[0], y=selected_columns[1], z=selected_columns[2],
                                        color=target_variable, color_continuous_scale='viridis', title="Gráfico 3D")
                    fig.update_layout(autosize=False, width=600, height=400, margin=dict(l=50, r=50, b=100, t=100, pad=4),
                                      paper_bgcolor="white",)
                    st.plotly_chart(fig)
                else:
                    st.warning('Para visualizar un gráfico 3D, selecciona al menos 3 columnas para el entrenamiento.')
                        # Predicciones en tiempo real
            if st.checkbox('Realizar una predicción en tiempo real'):
                input_dict = {}
                for column in selected_columns:
                    input_value = st.number_input(f'Introduce el valor de {column}')
                    input_dict[column] = input_value
                input_df = pd.DataFrame([input_dict])
                if preprocessing_option != 'Ninguno':
                    input_df = scaler.transform(input_df)
                prediction = knn.predict(input_df)
                st.write(f'La predicción es {prediction[0]}')

        else:
            st.warning('Por favor selecciona al menos una columna para entrenamiento y una variable objetivo.')

# Ejecución de la aplicación
if __name__ == "__main__":
    main()
