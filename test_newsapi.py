import requests

def test_newsapi(api_key, query="Apple"):
    """
    Test if the NewsAPI key is working correctly
    
    Args:
        api_key (str): Your NewsAPI key
        query (str): Search term
    """
    base_url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": api_key,
        "language": "en",
        "pageSize": 3  # Just get a few articles to test
    }
    
    try:
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            
            print(f"✅ API Key is working! Found {len(articles)} articles about '{query}'")
            print("\nSample headlines:")
            
            for i, article in enumerate(articles[:3], 1):
                print(f"{i}. {article['title']}")
                print(f"   Source: {article['source']['name']}")
                print(f"   Published: {article['publishedAt']}")
                print()
                
        elif response.status_code == 401:
            print("❌ API Key is invalid or unauthorized")
            print(f"Error message: {response.json().get('message', 'Unknown error')}")
            
        else:
            print(f"❌ Error: Status code {response.status_code}")
            print(f"Error message: {response.json().get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")

if __name__ == "__main__":
    api_key = input("Enter your NewsAPI key: ")
    query = input("Enter a search term (default: Apple): ") or "Apple"
    
    test_newsapi(api_key, query)