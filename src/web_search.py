import requests
from bs4 import BeautifulSoup


def perform_web_search(query: str, num_results: int = 5):
    search_url = f"https://www.google.com/search?q={query}&num={num_results}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    search_results = []
    for g in soup.find_all('div', class_='BNeawe vvjwJb AP7Wnd'):
        title = g.get_text()
        link = g.find_parent('a')['href']
        search_results.append((title, link))

    return search_results
