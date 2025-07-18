import requests
from textwrap import indent

def fetch_all_sets():
    url = "https://api.scryfall.com/sets"
    sets = {}
    while url:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        for s in data.get('data', []):
            sets[s['code']] = s['name']
        url = data.get('next_page')
    return sets

def pretty_print_dict(d):
    lines = ["all_sets = {"]
    for k, v in sorted(d.items()):
        lines.append(f'    "{k}": "{v}",')
    lines.append("}")
    print("\n".join(lines))

if __name__ == "__main__":
    all_sets = fetch_all_sets()
    pretty_print_dict(all_sets)
    print(f"\n# Total sets: {len(all_sets)}")