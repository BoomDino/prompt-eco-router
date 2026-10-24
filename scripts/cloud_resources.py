"""Read-only DigitalOcean resource inventory using the official PyDo SDK."""
import json
import os
import sys
from dotenv import load_dotenv


def main():
    load_dotenv()
    token = os.getenv('DIGITALOCEAN_TOKEN')
    if not token:
        sys.exit('Set DIGITALOCEAN_TOKEN to a control-plane token with Droplet read access.')
    try:
        from pydo import Client
    except ImportError:
        sys.exit('Install the optional SDK: pip install -r requirements-cloud.txt')
    try:
        client = Client(token=token)
        page = 1
        droplets = []
        while True:
            result = client.droplets.list(per_page=200, page=page)
            droplets.extend({k: d.get(k) for k in ('id', 'name', 'status', 'region', 'size_slug')}
                            for d in result.get('droplets', []))
            if not result.get('links', {}).get('pages', {}).get('next'):
                break
            page += 1
        print(json.dumps({'droplets': droplets}, indent=2))
    except Exception:
        sys.exit('DigitalOcean inventory failed. Check network access and token read permissions.')


if __name__ == '__main__':
    main()
