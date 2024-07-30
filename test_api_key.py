import requests

# Your RapidAPI key
RAPIDAPI_KEY = 'f9795e5735mshdedf63bc56d7fcbp131f52jsn30ca7a3d4f15'
RAPIDAPI_HOST_IMDB = 'imdb-api12.p.rapidapi.com'

def test_search_tv_show(title):
    url = f"https://{RAPIDAPI_HOST_IMDB}/title/find"
    headers = {
        'x-rapidapi-key': RAPIDAPI_KEY,
        'x-rapidapi-host': RAPIDAPI_HOST_IMDB
    }
    params = {'q': title}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=(30, 60))
        response.raise_for_status()
        print("Search TV show request successful")
        print(response.json())
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(response.text)
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")

# Test the function
test_search_tv_show("friends")
