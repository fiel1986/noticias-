import pandas as pd
import numpy as np
import requests
import joblib
import os
import random  
import smtplib 
from datetime import datetime, timedelta 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from flask import Flask, jsonify, request, render_template, flash, redirect, url_for
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

app = Flask(__name__)
app.secret_key = 'mpiz_pldm' 

# --- TUS API KEYS ---
NEWS_API_KEY = "a59d203e11ef71411494f2ecba624bdd"
OPENWEATHER_API_KEY = "b6bf80a069ca1aa7a0069519003fcfeb"
FIXER_API_KEY = "d8a1f13a1838921da6e5552613f1538a"

# --- CONFIGURACIÓN DE IA ---
BASE_LAT = 25.46
BASE_LON = -108.08

def obtener_modelo_ia(lat, lon, nombre_ciudad="Usuario"):
    """Carga o entrena el modelo para las coordenadas dadas."""
    modelo_id = f"{lat:.2f}_{lon:.2f}"
    ruta_temp = f'modelo_temp_{modelo_id}.pkl'
    ruta_lluvia = f'modelo_lluvia_{modelo_id}.pkl'
    
    if os.path.exists(ruta_temp) and os.path.exists(ruta_lluvia):
        print(f"🔄 Cargando IA para lat={lat}, lon={lon}...")
        return joblib.load(ruta_temp), joblib.load(ruta_lluvia)
    
    print(f"🚀 Entrenando nueva IA para lat={lat}, lon={lon}...")
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2022-01-01",
        "end_date": "2024-12-31",
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "relative_humidity_2m_max",
            "pressure_msl_mean", 
            "precipitation_sum"
        ],
        "timezone": "America/Mexico_City"
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            print("⚠️ No se pudieron descargar datos. Usando datos simulados.")
            return None, None
        
        data = resp.json()
        raw_daily = data['daily']
        
        df = pd.DataFrame({
            'fecha': raw_daily['time'],
            'temp_max': raw_daily['temperature_2m_max'],
            'temp_min': raw_daily['temperature_2m_min'],
            'humedad_max': raw_daily['relative_humidity_2m_max'],
            'presion': raw_daily['pressure_msl_mean'], 
            'lluvia': raw_daily['precipitation_sum']   
        })
        
        df['fecha'] = pd.to_datetime(df['fecha'])
        
        df['mes'] = df['fecha'].dt.month
        df['dia_semana'] = df['fecha'].dt.dayofweek
        df['dia_anio'] = df['fecha'].dt.dayofyear
        df['estacion'] = df['mes'].apply(lambda m: 0 if m in [12,1,2] else (1 if m in [3,4,5] else (2 if m in [6,7,8] else 3)))
        
        df['temp_siguiente'] = df['temp_max'].shift(-1)
        df['llueve_siguiente'] = (df['lluvia'].shift(-1) > 0.5).astype(int)
        
        df = df.dropna()
        
        if len(df) < 100:
            return None, None

        features = ['temp_max', 'temp_min', 'humedad_max', 'presion', 'lluvia', 'mes', 'dia_semana', 'dia_anio', 'estacion']
        X = df[features].values
        y_temp = df['temp_siguiente'].values
        y_lluvia = df['llueve_siguiente'].values

        modelo_temp = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
        modelo_lluvia = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
        
        modelo_temp.fit(X, y_temp)
        modelo_lluvia.fit(X, y_lluvia)
        
        joblib.dump(modelo_temp, ruta_temp)
        joblib.dump(modelo_lluvia, ruta_lluvia)
        
        return modelo_temp, modelo_lluvia
        
    except Exception as e:
        print(f"❌ Error entrenando IA: {e}")
        return None, None

def predecir_ia_con_coord(lat, lon, modelo_temp, modelo_lluvia, ciudad="Usuario"):
    """Realiza la predicción usando los modelos entrenados y datos reales de hoy."""
    if not modelo_temp or not modelo_lluvia:
        return None
    
    try:
        # 1. Obtener datos REALES de hoy
        url_clima = f"http://api.openweathermap.org/data/2.5/weather?q={ciudad}&appid={OPENWEATHER_API_KEY}&units=metric"
        
        response = requests.get(url_clima, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            temp_max_hoy = round(data['main']['temp_max'])
            temp_min_hoy = round(data['main']['temp_min'])
            humedad_max_hoy = data['main']['humidity']
            presion_hoy = data['main']['pressure']
            condition_code = data['weather'][0]['id']
            
            va_llover_hoy = 200 <= condition_code <= 599
            lluvia_hoy = 1.0 if va_llover_hoy else 0.0
            
        else:
            print("⚠️ No se pudieron obtener datos de hoy, usando valores promedio.")
            temp_max_hoy, temp_min_hoy, humedad_max_hoy, presion_hoy, lluvia_hoy = 28.0, 22.0, 65.0, 1013.0, 0.0

        hoy = pd.Timestamp.now()
        
        df_hoy = pd.DataFrame({
            'temp_max': [temp_max_hoy],
            'temp_min': [temp_min_hoy],
            'humedad_max': [humedad_max_hoy],
            'presion': [presion_hoy],
            'lluvia': [lluvia_hoy],
            'mes': [hoy.month],
            'dia_semana': [hoy.dayofweek],
            'dia_anio': [hoy.dayofyear],
            'estacion': [0 if hoy.month in [12,1,2] else (1 if hoy.month in [3,4,5] else (2 if hoy.month in [6,7,8] else 3))]
        })
        
        features = ['temp_max', 'temp_min', 'humedad_max', 'presion', 'lluvia', 'mes', 'dia_semana', 'dia_anio', 'estacion']
        X_pred = df_hoy[features].values
        
        pred_temp = modelo_temp.predict(X_pred)[0]
        prob_lluvia = modelo_lluvia.predict_proba(X_pred)[0, 1]
        va_llover = "Sí" if prob_lluvia > 0.5 else "No"
        
        return {
            "temperatura_max_mañana": round(pred_temp, 1),
            "probabilidad_lluvia": round(prob_lluvia * 100, 1),
            "va_llover": va_llover,
            "coordenadas": {"lat": lat, "lon": lon}
        }
        
    except Exception as e:
        print(f"❌ Error en predecir_ia_con_coord: {e}")
        return None

@app.route('/api/prediccion-ia-geo', methods=['GET'])
def prediccion_ia_geo():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    
    if lat is None or lon is None:
        return jsonify({'status': 'error', 'mensaje': 'Faltan coordenadas lat/lon'}), 400
    
    ciudad = request.args.get('ciudad', 'Usuario')
    
    # Entrenar/Cargar modelo
    modelo_temp, modelo_lluvia = obtener_modelo_ia(lat, lon)
    
    if not modelo_temp:
        return jsonify({
            'status': 'error', 
            'mensaje': 'No se pudo entrenar el modelo (datos insuficientes o error de API).'
        }), 500
    
    # Llamada correcta a la función existente
    resultado = predecir_ia_con_coord(lat, lon, modelo_temp, modelo_lluvia, ciudad)
    
    if resultado:
        return jsonify({
            'status': 'success',
            'prediccion': resultado
        })
    else:
        return jsonify({'status': 'error', 'mensaje': 'Error en la predicción'}), 500

# --- VALIDACIÓN DE API KEY ---
def verificar_api_key_clima():
    if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == 'demo_mode':
        return False
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q=London&appid={OPENWEATHER_API_KEY}"
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except:
        return False

API_KEY_VALIDA = verificar_api_key_clima()
print(f"Modo API Clima: {'ACTIVO' if API_KEY_VALIDA else 'DEMO'}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/finanzas')
def finanzas():
    return render_template('finanzas.html')

@app.route('/contacto', methods=['GET', 'POST'])
def contacto():
    if request.method == 'POST':
        nombre = request.form.get('name')
        email = request.form.get('email')
        telefono = request.form.get('phone')
        asunto = request.form.get('subject')
        mensaje = request.form.get('message')

        if not nombre or not email or not mensaje:
            flash('Por favor completa los campos obligatorios.', 'error')
            return redirect(url_for('contacto'))

        try:
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            sender_email = "fiel10071986@gmail.com"
            # ⚠️ IMPORTANTE: Reemplaza esto con tu contraseña de aplicación de Google
            sender_password = "TU_PASSWORD_DE_APLICACION" 
            recipient_email = "fiel10071986@gmail.com"

            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = recipient_email
            msg['Subject'] = f"Nuevo mensaje de contacto: {asunto}"
            
            body = f"Nombre: {nombre}\nEmail: {email}\nTeléfono: {telefono}\nAsunto: {asunto}\n\nMensaje: {mensaje}"
            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            flash('¡Mensaje enviado con éxito!', 'success')
        except Exception as e:
            print(f"Error al enviar email: {e}")
            flash('Error al enviar el mensaje. Revisa la consola.', 'error')

        return redirect(url_for('contacto'))

    return render_template('contacto.html')

@app.route('/api/weather')
def obtener_clima():
    ciudad = request.args.get('q', 'Mexico City')
    
    if not API_KEY_VALIDA:
        return generar_clima_demo(ciudad)
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={ciudad}&appid={OPENWEATHER_API_KEY}&units=metric&lang=es"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            clima_info = {
                'ciudad': data['name'],
                'pais': data['sys']['country'],
                'temperatura': round(data['main']['temp']),
                'sensacion_termica': round(data['main']['feels_like']),
                'humedad': data['main']['humidity'],
                'presion': data['main']['pressure'],
                'viento': round(data['wind']['speed'] * 3.6, 1),
                'descripcion': data['weather'][0]['description'].capitalize(),
                'icono': data['weather'][0]['icon'],
                'maxima': round(data['main']['temp_max']),
                'minima': round(data['main']['temp_min']),
                'condicion': data['weather'][0]['main'].lower(),
                'lat': data['coord']['lat'],
                'lon': data['coord']['lon']
            }
            return jsonify({
                'status': 'success',
                'clima': clima_info,
                'timestamp': datetime.now().isoformat(),
                'modo': 'api_real'
            })
        else:
            return generar_clima_demo(ciudad)
    except Exception as e:
        print(f"Excepción API Clima: {str(e)}")
        return generar_clima_demo(ciudad)

@app.route('/api/forecast')
def obtener_pronostico():
    ciudad = request.args.get('q', 'Mexico City')
    
    if not API_KEY_VALIDA:
        return generar_pronostico_demo(ciudad)
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={ciudad}&appid={OPENWEATHER_API_KEY}&units=metric&lang=es"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            pronostico_dias, pronostico_horas = procesar_pronostico_completo(data['list'])
            
            return jsonify({
                'status': 'success',
                'ciudad': data['city']['name'],
                'pais': data['city']['country'],
                'coordenadas': data['city']['coord'],
                'pronostico_dias': pronostico_dias,
                'pronostico_horas': pronostico_horas,
                'modo': 'api_real'
            })
        else:
            return generar_pronostico_demo(ciudad)
    except Exception as e:
        print(f"Excepción API Pronóstico: {str(e)}")
        return generar_pronostico_demo(ciudad)
@app.route('/api/gnews')
def obtener_gnews():
    # Obtener parámetros de la URL
    category = request.args.get('category', 'general')
    country = request.args.get('country', 'mx')
    
    # URL de la API de noticias
    url = f"https://gnews.io/api/v4/top-headlines?category={category}&lang=es&country={country}&max=5&apikey={NEWS_API_KEY}"
    
    print(f"📰 Solicitando noticias a: {url}") # Para depuración

    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'articles' in data:
                print(f"✅ Noticias cargadas correctamente: {len(data['articles'])} artículos.")
                return jsonify({
                    'status': 'success',
                    'articles': data['articles'],
                    'modo': 'api_real'
                })
            else:
                print("⚠️ La API respondió OK pero no hay artículos.")
                return generar_noticias_demo(category, country)
        else:
            print(f"❌ Error API Noticias: Código {response.status_code}")
            # Imprimir el mensaje de error de la API para saber qué pasa (ej. API Key inválida)
            print(f"Respuesta API: {response.text}")
            return generar_noticias_demo(category, country)
            
    except Exception as e:
        print(f"❌ Excepción en obtener_gnews: {str(e)}")
        return generar_noticias_demo(category, country)

def generar_noticias_demo(category, country):
    """Genera noticias falsas si la API falla."""
    noticias = [
        {
            'title': f"Noticia destacada de {category} en {country}",
            'description': 'Esta es una noticia de ejemplo generada automáticamente porque la API falló o no hay datos.',
            'url': '#',
            'image': 'https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=400&h=200&fit=crop',
            'source': {'name': 'Noticias Demo'},
            'publishedAt': datetime.now().isoformat(),
            'content': 'Contenido de ejemplo para demostración.'
        },
        {
            'title': 'Avances tecnológicos recientes',
            'description': 'Nuevos desarrollos en el mercado global de tecnología.',
            'url': '#',
            'image': 'https://images.unsplash.com/photo-1519389950473-47ba0277781c?w=400&h=200&fit=crop',
            'source': {'name': 'Tech News'},
            'publishedAt': datetime.now().isoformat(),
            'content': 'Más contenido de ejemplo...'
        },
        {
            'title': 'Economía mundial en alza',
            'description': 'Los mercados muestran tendencias positivas.',
            'url': '#',
            'image': 'https://images.unsplash.com/photo-1611974765270-ca1258634369?w=400&h=200&fit=crop',
            'source': {'name': 'Finanzas Demo'},
            'publishedAt': datetime.now().isoformat(),
            'content': 'Detalle financiero...'
        }
    ]
    return jsonify({
        'status': 'success',
        'articles': noticias,
        'modo': 'demo'
    })

@app.route('/api/clima_coords')
def obtener_clima_coords():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    
    if not lat or not lon:
        return jsonify({'status': 'error', 'mensaje': 'Faltan parámetros lat o lon'}), 400

    if not API_KEY_VALIDA:
        return generar_clima_demo("Tu ubicación")

    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric&lang=es"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            clima_info = {
                'ciudad': data['name'],
                'pais': data['sys']['country'],
                'temperatura': round(data['main']['temp']),
                'sensacion_termica': round(data['main']['feels_like']),
                'humedad': data['main']['humidity'],
                'viento': round(data['wind']['speed'] * 3.6, 1),
                'descripcion': data['weather'][0]['description'].capitalize(),
                'icono': data['weather'][0]['icon'],
                'condicion': data['weather'][0]['main'].lower(),
                'lat': data['coord']['lat'],
                'lon': data['coord']['lon']
            }
            return jsonify({'status': 'success', 'clima': clima_info})
        else:
            return generar_clima_demo("Error")
    except Exception as e:
        print(f"Excepción API Coords: {str(e)}")
        return generar_clima_demo("Error")

# ... (otras rutas como /api/gnews, /api/moneda, etc.) ...

def generar_clima_demo(ciudad):
    condiciones = ['clear', 'clouds', 'rain', 'drizzle']
    condicion = random.choice(condiciones)
    temp = random.randint(15, 30)
    emojis = {'clear': '01d', 'clouds': '02d', 'rain': '10d', 'drizzle': '09d'}
    
    return jsonify({
        "status": "success",
        "clima": {
            "ciudad": ciudad.title(),
            "pais": "Demo",
            "temperatura": temp,
            "sensacion_termica": temp + random.randint(-2, 2),
            "humedad": random.randint(40, 80),
            "presion": random.randint(1000, 1020),
            "viento": round(random.uniform(5.0, 20.0), 1),
            "descripcion": "Clima de ejemplo",
            "icono": emojis.get(condicion, '01d'),
            "maxima": temp + random.randint(2, 5),
            "minima": temp - random.randint(3, 8),
            "condicion": condicion,
            "lat": 19.4326,
            "lon": -99.1332
        },
        "modo": "demo"
    })

def generar_pronostico_demo(ciudad):
    hoy = datetime.now()
    dias = []
    for i in range(5):
        fecha = hoy.date() + timedelta(days=i)
        temp = random.randint(18, 28)
        condicion = random.choice(['clear', 'clouds', 'rain'])
        dias.append({
            'fecha': fecha.isoformat(),
            'fecha_formateada': fecha.strftime('%Y-%m-%d'),
            'dia_semana': fecha.strftime('%A'),
            'temperatura': temp,
            'maxima': temp + 3,
            'minima': temp - 3,
            'condicion': condicion,
            'descripcion': 'Clima variable',
            'icono': '01d',
            'es_hoy': i == 0,
            'horas_detalle': []
        })
    return jsonify({
        'status': 'success',
        'pronostico_dias': dias,
        'modo': 'demo'
    })

def generar_noticias_demo(category, country):
    noticias = [
        {
            'title': f"Noticia de {category} en {country}",
            'description': 'Esta es una noticia de ejemplo generada automáticamente.',
            'url': '#',
            'image': 'https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=400&h=200&fit=crop',
            'source': {'name': 'Noticias Demo'},
            'publishedAt': datetime.now().isoformat(),
            'content': 'Contenido de ejemplo...'
        },
        {
            'title': 'Avances tecnológicos recientes',
            'description': 'Nuevos desarrollos en el mercado.',
            'url': '#',
            'image': 'https://images.unsplash.com/photo-1519389950473-47ba0277781c?w=400&h=200&fit=crop',
            'source': {'name': 'Tech News'},
            'publishedAt': datetime.now().isoformat(),
            'content': 'Más contenido...'
        }
    ]
    return jsonify({
        'status': 'success',
        'articles': noticias,
        'modo': 'demo'
    })

def procesar_pronostico_completo(datos_pronostico):
    dias = {}
    horas = []
    
    for item in datos_pronostico:
        dt_txt = item['dt_txt']
        fecha_obj = datetime.strptime(dt_txt, "%Y-%m-%d %H:%M:%S")
        dia_key = fecha_obj.strftime("%Y-%m-%d")
        
        if dia_key not in dias:
            dias[dia_key] = {
                'fecha': dia_key,
                'fecha_formateada': fecha_obj.strftime('%Y-%m-%d'),
                'dia_semana': fecha_obj.strftime('%A'),
                'temperatura': item['main']['temp'],
                'maxima': item['main']['temp_max'],
                'minima': item['main']['temp_min'],
                'condicion': item['weather'][0]['main'].lower(),
                'descripcion': item['weather'][0]['description'],
                'icono': item['weather'][0]['icon'],
                'es_hoy': fecha_obj.date() == datetime.now().date(),
                'horas_detalle': []
            }
        
        dias[dia_key]['maxima'] = max(dias[dia_key]['maxima'], item['main']['temp_max'])
        dias[dia_key]['minima'] = min(dias[dia_key]['minima'], item['main']['temp_min'])
          
        horas.append({
            'fecha': dt_txt,
            'hora': fecha_obj.strftime("%H:%M"),
            'temperatura': round(item['main']['temp'], 1),
            'icono': item['weather'][0]['icon'],
            'descripcion': item['weather'][0]['description']
        })   

    lista_dias = sorted(dias.values(), key=lambda x: x['fecha'])
    return lista_dias, horas

if __name__ == '__main__':
    print("Iniciando despegue 1,2,3...")

    app.run(debug=True, port=5000)
