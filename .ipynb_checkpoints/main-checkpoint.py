import os
import uuid
import base64
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

def get_token():
    """Получаем токен авторизации"""
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    
    # Берём Client ID и Client Secret из .env
    client_id = os.getenv("GIGA_CLIENT_ID")
    client_secret = os.getenv("GIGA_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        print("Ошибка: в .env нет GIGA_CLIENT_ID / GIGA_CLIENT_SECRET")
        return None
    
    # Кодируем в формате client_id:client_secret -> Base64
    auth_string = f"{client_id}:{client_secret}"
    auth_base64 = base64.b64encode(auth_string.encode()).decode()
    
    headers = {
        "Authorization": f"Basic {auth_base64}",
        "RqUID": str(uuid.uuid4()),
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    response = requests.post(
        url, 
        headers=headers, 
        data={"scope": os.getenv("GIGA_SCOPE", "GIGACHAT_API_PERS")}, 
        verify=False
    )
    
    if response.status_code != 200:
        print(f"Ошибка авторизации: {response.status_code}")
        print(f"Ответ сервера: {response.text}")
        return None
    
    return response.json()["access_token"]

def ask_gigachat(question):
    token = get_token()
    if not token:
        return "Не удалось получить токен. Проверь .env"
    
    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    data = {
        "model": "GigaChat",
        "messages": [
            {"role": "system", "content": "Ты строгий эксперт. Отвечай фактами. Если не знаешь, напиши: 'Я не знаю'."},
            {"role": "user", "content": question}
        ],
        "temperature": 0.2
    }
    
    response = requests.post(url, headers=headers, json=data, verify=False)
    
    if response.status_code != 200:
        print(f"Ошибка запроса: {response.status_code}")
        print(f"Ответ: {response.text}")
        return "Ошибка при запросе"
    
    result = response.json()
    print(f"\n[Токены] Использовано: {result['usage']['total_tokens']}")
    return result["choices"][0]["message"]["content"]

user_input = input("Задайте вопрос: ")
print(f"\nОтвет: {ask_gigachat(user_input)}")
