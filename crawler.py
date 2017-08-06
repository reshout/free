import datetime
import csv
from random import shuffle
from pandas_datareader import data
from pandas_datareader._utils import RemoteDataError


def save_item(market, code, name, days):
    end = datetime.datetime.now().date()
    start = end - datetime.timedelta(days)
    path = './{}/{}-{}.csv'.format(market, code, name)
    stock_data = data.DataReader("{}.{}".format(code, 'KS' if market is 'KOSPI' else 'KQ'), 'yahoo', start, end)
    stock_data.to_csv(path)


def save_items(market, days=365):
    items = read_items(market) 
    shuffle(items)
    total = len(items)
    done = 0
    while len(items) > 0:
        item = items.pop()
        code = item[0]
        name = item[1]
        try:
            print('[{}/{}]'.format(done, total), 'Processing {} {}'.format(code, name), end=' ... ')
            save_item(market,code, name, days)
            done += 1
            print('Success')
        except RemoteDataError:
            print('Failed')
            items.insert(0, item)


def read_items(market):
    items = []
    for csv_row in csv.reader(open('{}.csv'.format(market), 'r')):
        if len(csv_row[0]) == 0:
            continue
        name = csv_row[1]
        code = csv_row[2]
        items.append((code, name))
    return items


save_items('KOSPI')
