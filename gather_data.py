# Every five minutes, this script will use the coinpaprika api key to get the current BTC and ETH prices and save them to respective csv files
from coinpaprika import client as Coinpaprika
import time
from datetime import datetime

client = Coinpaprika.Client()


def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)

btc_old = 0
eth_old = 0
while True:
    btc = client.ticker("btc-bitcoin")["quotes"]["USD"]["price"]
    eth = client.ticker("eth-ethereum")["quotes"]["USD"]["price"]

    date_object = str(datetime.now().date())
    hour = datetime.now().hour
    minute = datetime.now().minute

    v = minute % 5
    minute -= v
    new_time = str(hour) + ":" + str(minute)
    btc_obj = date_object + "," + new_time + "," + str(btc)
    eth_obj = date_object + "," + new_time + "," + str(eth)

    if btc == btc_old or eth == eth_old:
        time.sleep(30)
    else:
        append_new_line('btc_values.txt', btc_obj)
        append_new_line('eth_values.txt', eth_obj)
        btc_old = btc
        eth_old = eth
        time.sleep(300)
