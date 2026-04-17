import requests

BASE_URL = "http://127.0.0.1:8000"

def test_recommendation_existing_user():
    print("--- Testing Existing User (ID: 1) ---")
    payload = {"user_id": 1, "num_rec": 5}
    response = requests.post(f"{BASE_URL}/recommend", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        latency = response.headers.get("X-Process-Time")
        print(f"Status: SUCCESS")
        print(f"Latency Header (X-Process-Time): {latency}s")
        print(f"Recommendations: {len(data['recs'])} items returned.")
        for rec in data['recs'][:3]:
            print(f"  - MovieID: {rec['movieId']}, Score: {rec['score']:.4f}")
    else:
        print(f"Status: FAILED ({response.status_code})")
        print(response.text)

def test_recommendation_cold_start():
    print("\n--- Testing Cold Start (New User ID: 99999) ---")
    payload = {"user_id": 99999, "num_rec": 5}
    response = requests.post(f"{BASE_URL}/recommend", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        latency = response.headers.get("X-Process-Time")
        print(f"Status: SUCCESS (Fallback logic active)")
        print(f"Latency Header (X-Process-Time): {latency}s")
        print(f"Recommendations: {len(data['recs'])} items returned.")
    else:
        print(f"Status: FAILED ({response.status_code})")

if __name__ == "__main__":
    try:
        test_recommendation_existing_user()
        test_recommendation_cold_start()
    except Exception as e:
        print(f"Error connecting to server: {e}")
        print("Make sure 'uvicorn api:app --reload' is running in another terminal.")
