from flask import Flask, render_template, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import joblib
from sqlalchemy import create_engine
import mysql.connector
import requests
import re

app = Flask(__name__)

# Cargar el modelo, los label encoders y el scaler
kmeans, label_encoders, scaler = joblib.load('clustering_model.pkl')

# Configurar la conexión a la base de datos
db_user = '*****'
db_password = '*****'
db_host = '*****'
db_port = '*****'
db_name = '*****'
DATABASE_URI = f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
engine = create_engine(DATABASE_URI)

@app.route('/predict_and_update', methods=['GET', 'POST'])
def predict_and_update():
    try:
        # Obtener el último usuario
        query = "SELECT * FROM Users ORDER BY id DESC LIMIT 1;"
        user_new1 = pd.read_sql(query, engine)
        
        if user_new1.empty:
            return jsonify({"error": "No se encontró ningún usuario"}), 404
        
        # Transformar las columnas categóricas
        user_new = user_new1[['rol', 'puesto_trabajo', 'pais']]
        
        for column in ['rol', 'puesto_trabajo', 'pais']:
            if column in label_encoders:
                user_new[column] = label_encoders[column].transform(user_new[column].astype(str))
            else:
                return jsonify({"error": f"Categoría desconocida en la columna {column}"}), 400

        # Normalizar los datos
        user_new_scaled = scaler.transform(user_new)

        # Predecir el cluster
        cluster = kmeans.predict(user_new_scaled)[0]

        # Actualizar el dataframe con el nuevo cluster
        user_new1['cluster'] = cluster

        # Extraer los valores de cluster e id
        cluster_value = int(user_new1['cluster'][0])
        id_value = int(user_new1['id'][0])

        # Establecer conexión a la base de datos
        conn = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name
        )

        # Crear cursor para ejecutar consultas
        cursor = conn.cursor()

        # Construir la consulta SQL de actualización
        update_query = """
            UPDATE Users 
            SET cluster = %s 
            WHERE id = %s;
        """

        # Ejecutar la consulta
        cursor.execute(update_query, (cluster_value, id_value))
        
        # Commit para aplicar los cambios en la base de datos
        conn.commit()

        # Verificar si la fila se ha actualizado correctamente
        cursor.execute("SELECT cluster FROM Users WHERE id = %s", (id_value,))
        updated_cluster = cursor.fetchone()[0]
        
        if updated_cluster == cluster_value:
            # Enviar la información actualizada al endpoint
            endpoint_url = 'https://e-learning-experience.onrender.com/users'
            response = requests.post(endpoint_url, json=user_new1.to_dict(orient='records')[0])
            if response.status_code == 200:
                return render_template('result.html', id=id_value, cluster=cluster_value)
            else:
                return jsonify({"error": f"Error al enviar los datos al endpoint: {response.text}"}), 500
        else:
            return jsonify({"error": "La fila no se actualizó correctamente."}), 500

    except Exception as e:
        if 'conn' in locals() and conn:
            conn.rollback()  # Revertir cambios en caso de error
        return jsonify({"error": str(e)}), 500

    finally:
        # Cerrar cursor y conexión
        if 'cursor' in locals() and cursor is not None:
            cursor.close()
        if 'conn' in locals() and conn is not None:
            conn.close()

if __name__ == '__main__':
    app.run(debug=True)
