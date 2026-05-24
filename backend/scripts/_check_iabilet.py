"""Debug: find all ?widget= occurrences and check Playwright availability."""
import sys
import re
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

sys.stdout.reconfigure(encoding="utf-8")

ua = UserAgent()
session = requests.Session()
session.headers.update({
    "User-Agent": ua.random,
    "Accept-Language": "ro-RO,ro;q=0.9",
    "Referer": "https://www.iabilet.ro/",
})

r = session.get("https://www.iabilet.ro/bilete-in-bucuresti/?page=1", timeout=15)
text = r.text

# Simple string search for widget param
idx = 0
widget_occurrences = []
while True:
    pos = text.find("widget=", idx)
    if pos == -1:
        break
    widget_occurrences.append(text[max(0, pos-5):pos+300])
    idx = pos + 1

print(f"'widget=' occurrences: {len(widget_occurrences)}")
for w in widget_occurrences[:5]:
    print("  ...", repr(w[:200]))
    print()

# Find all function calls ending with ) that contain a URL-like string
url_in_js = re.findall(r'"\\/\?widget=[^"]{30,}"', text)
print(f"\n\\/\\?widget= in quotes: {len(url_in_js)}")
for u in url_in_js[:5]:
    print(" ", u[:200])

# Check for playwright
try:
    import playwright
    print("\nPlaywright: AVAILABLE")
except ImportError:
    print("\nPlaywright: not installed")

# Also check selenium
try:
    import selenium
    print("Selenium: AVAILABLE")
except ImportError:
    print("Selenium: not installed")
