import requests
from bs4 import BeautifulSoup
from datetime import date

def scrape_isw(target_date: date) -> dict:
    url = f"https://www.understandingwar.org/backgrounder/russian-offensive-campaign-assessment-{target_date.strftime('%B-%d-%Y').lower()}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "lxml")
    title = soup.find("h1").text.strip() if soup.find("h1") else "No title"
    return {"date": str(target_date), "url": url, "title": title}

if __name__ == "__main__":
    result = scrape_isw(date.today())
    print(result)
