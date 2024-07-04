from flask import Flask, request, render_template, redirect, url_for
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
from google.oauth2 import service_account
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# Initialize Vertex AI
credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
credentials = service_account.Credentials.from_service_account_file(credentials_path)
vertexai.init(project="equipo-2-427316", location="europe-west1", credentials=credentials)
model = GenerativeModel("gemini-1.5-flash-001")

generation_config = {
    "max_output_tokens": 400,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')

DATABASE_URI = f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
engine = create_engine(DATABASE_URI,pool_recycle=3600, pool_pre_ping=True)
Session = sessionmaker(bind=engine)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video', methods=['POST'])
def process_video():
    if 'video' not in request.files or 'youtube_link' not in request.form or 'event_id' not in request.form:
        return redirect(url_for('index'))
    
    video_file = request.files['video']
    youtube_link = request.form['youtube_link']
    event_id = request.form['event_id']

    video_data = video_file.read()
    video_base64 = base64.b64encode(video_data).decode('utf-8')

    video_part = Part.from_data(mime_type="video/mp4", data=base64.b64decode(video_base64))
    text1 = """Proporciona un resumen claro y conciso de 3-5 parrafos con puntos importantes y maximo 350 tokens.
    Quiero un texto bonito, bien estructurado y listo para enviar a los clientes.
    IMPORTANTE: No escribas introduccion tipo: \"Claro, aqui tienes tu resumen:\""""

    responses = model.generate_content([video_part, text1], generation_config=generation_config, safety_settings=safety_settings, stream=True)
    summary = ""
    for response in responses:
        summary += response.text
    
    # Insert the data into the database
    created_at = datetime.now()
    updated_at = datetime.now()
    query = text("INSERT INTO Summaries (event_id, youtube_link, summary, createdAt, updatedAt) VALUES (:event_id, :youtube_link, :summary, :created_at, :updated_at)")
    params = {
        "event_id": event_id,
        "youtube_link": youtube_link,
        "summary": summary,
        "created_at": created_at,
        "updated_at": updated_at
    }

    session = Session()
    try:
        session.execute(query, params)
        session.commit()  # Commit the transaction to save the changes
    except Exception as e:
        session.rollback()  # Rollback the transaction on error
        print(f"Error: {e}")
    finally:
        session.close()

    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)