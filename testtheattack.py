import requests

url = 'http://127.0.0.1:5000/simulate_attack'
payload = {'payload': '1"))) or 6793=(select 6793 from pg_sleep(5)) and ((("azmf"="azmf'}

try:
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        try:
            result = response.json()
            print(result)  # Print or process the JSON response
        except ValueError as e:
            print(f"Error decoding JSON: {e}")
    else:
        print(f"Error: {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"Request error: {e}")

