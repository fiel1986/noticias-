from flask import Flask, render_template, request, flash, redirect, url_for
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import requests

app = Flask(__name__)
app.secret_key = 'mpiz pldm wpxi gngn' 
NEWS_API_KEY = "a59d203e11ef71411494f2ecba624bdd"
OPENWEATHER_API_KEY = "b6bf80a069ca1aa7a0069519003fcfeb"
FIXER_API_KEY = "d8a1f13a1838921da6e5552613f1538a"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/finanzas')
def finanzas():
    return render_template('finanzas.html')

# --- API MONEDAS ---
@app.route('/api/moneda')
def obtener_moneda():
    base = request.args.get("base", "USD")      
    simbolos = request.args.get("simbolos", "MXN,EUR")  
    try:
        url = f"https://data.fixer.io/api/latest?access_key={FIXER_API_KEY}&base={base}&symbols={simbolos}"
        response = requests.get(url)
        data = response.json()
        if not data.get("success"):
            return {"status": "error", "mensaje": data.get("error", {}).get("info", "Error en Fixer")}
        return {"status": "ok", "base": base, "rates": data.get("rates", {}), "fecha": data.get("date")}
    except Exception as e:
        return {"status": "error", "mensaje": str(e)}

# --- API CLIMA ---
@app.route('/api/weather')
def weather_proxy():
    city = request.args.get("q", "Mexico City")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric&lang=es"
    return requests.get(url).json()

@app.route('/api/forecast')
def forecast_proxy():
    city = request.args.get("q", "Mexico City")
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPENWEATHER_API_KEY}&units=metric&lang=es"
    return requests.get(url).json()

# --- API NOTICIAS ---
@app.route('/api/gnews')
def gnews_proxy():
    category = request.args.get("category", "general")
    country = request.args.get("country", "mx")
    url = f"https://gnews.io/api/v4/top-headlines?category={category}&lang=es&country={country}&max=10&token={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        # Formateamos para que tu JS reciba lo que espera
        noticias = [
            {
                "title": a.get("title"),
                "description": a.get("description"),
                "url": a.get("url"),
                "image": a.get("image"),
                "source": {"name": a.get("source", {}).get("name", "GNews")},
                "publishedAt": a.get("publishedAt"),
                "content": a.get("content")
            }
            for a in data.get("articles", [])
        ]
        return {"status": "ok", "articles": noticias}
    except Exception as e:
        return {"status": "error", "mensaje": str(e)}

# --- CONTACTO ---
@app.route('/contacto', methods=['GET', 'POST']) 
def contacto():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        phone = request.form.get("phone") 
        message = request.form.get("message")
        if not name or not email or not message:
            flash('Por favor completa todos los campos', 'error')
            return render_template('contacto.html')
        try:
            send_email(name, email, phone, message)
            flash('¡Mensaje enviado!', 'success')
            return redirect(url_for('contacto'))
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
            return render_template('contacto.html')
    return render_template('contacto.html')

def send_email(name, email, phone, message):
    sender_email = "fiel10071986@gmail.com"  
    sender_password = "mpiz pldm wpxi gngn" 
    recipient_email = "fiel10071986@gmail.com" 
    subject = "Nuevo mensaje de contacto"
    body = f"Nombre: {name}\nCorreo: {email}\nTel: {phone}\nMensaje: {message}"
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)  

if __name__ == '__main__':
    app.run(debug=True, port=5500)
