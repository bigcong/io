import requests

url = 'https://www.okex.com/v2/c2c-open/tradingOrders/group?digitalCurrencySymbol=eth&legalCurrencySymbol=cny&best=0&exchangeRateLevel=0&paySupport=0'
headers = {

    'accept': 'application/json',
    'accept-encoding': 'gzip, deflate, br',
    'accept-language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
    'authorization': 'eyJhbGciOiJIUzUxMiJ9.eyJqdGkiOiI2ZDI3Zjk0MC0xNDVmLTQ3YzEtOGU1OC05NjM5NGE2Y2ViOTZiUVlCIiwidWlkIjoiaGhDQkxDUUxwd0ZDR3BKOE8zdTFXZz09Iiwic3ViIjoiMTczKioqMTk4OSIsImVtbCI6ImN1aWNvbmcwNzE1QGdtYWlsLmNvbSIsInN0YSI6MCwibWlkIjowLCJpYXQiOjE1MzQyMDc4NzAsImV4cCI6MTUzNDgxMjY3MCwiYmlkIjowLCJkb20iOiJ3d3cub2tleC5jb20iLCJpc3MiOiJva2NvaW4ifQ.K650yMUYo_6mXe7ZQAKm6UcNiupuUxlBguSFWHo3RNFy7XMaAFiae_u5HcM-bTAsUlq0pSTYugLQXjLpcG5yaA',
    'content-type': 'application/json',
    'cookie': '__cfduid=dbd1b6e104fbe36917bfd67151c7de2ef1527987774; _ga=GA1.2.1036434625.1527987777; perm=4FA0831DD6F93844AD9D8C81986C8C83; locale=zh_CN; currencyId=2; currency=ETH; Hm_lvt_b4e1f9d04a77cfd5db302bc2bcc6fe45=1532447896; lp=/future/trade; first_ref=https://www.okex.com/; _gid=GA1.2.1834169193.1534169762; isLogin=1; kycNationality=CN; product=gto_eth; ref=https://www.okex.com/futureTrade/futureIndex?currencyId=2&index=1; Hm_lpvt_b4e1f9d04a77cfd5db302bc2bcc6fe45=1534695984',
    'referer': 'https://www.okex.com/fiat/c2c',
    'ser-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36'
}

json = requests.get(url, headers=headers).json()

print("买一数量:", json['data']['buyTradingOrders'][0]['availableAmount'])
print("买一价格:", json['data']['buyTradingOrders'][0]['exchangeRate'])
print("买一最小金额:", json['data']['buyTradingOrders'][0]['minPlacePrice'])

sell = json['data']['sellTradingOrders']
last = sell[len(sell) - 1]

print("卖一数量:", last['availableAmount'])
print("卖-价格:", last['exchangeRate'])
print("卖一最小金额:", last['minPlacePrice'])

print()
