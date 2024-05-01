import urllib.request, json
from config import Config
import os
import pandas as pd
import datetime as dt
from utils import Utils

config = Config()
utils = Utils()

class AlphaVantage:
    
    def __init__(self):
        self.API_KEY = config.get_property("ALPHAVANTAGE_API_KEY")
        self.BASE_URL_TEMPLATE = config.get_property("ALPHAVANTAGE_URL")
        self.save_file_name_template = config.get_property("save_file_name_template")
        

    
    def __hit_api_and_get_data(self, URL):
        
        try:
            with urllib.request.urlopen(URL) as url:
                data = json.loads(url.read().decode())
        except Exception as e:
            print(e)
        
        return data
    
    
    
    def __save_data_frame(self, df, save_file_name):
        df.to_csv(save_file_name)
        
        
        
    
    def get_ticker_data(self, ticker):
        
        URL_WITH_TICKER = self.BASE_URL_TEMPLATE.format(
                            time_series=config.get_property("time_series_type", "WEEKLY").upper(),
                            symbol=ticker, 
                            apikey=self.API_KEY
                        )
        
        
        
        save_file_name = self.save_file_name_template.format(ticker=ticker, date=utils.get_datetime(), time_series_type=config.get_property("time_series_type"))
        
        
        if not utils.does_file_exist(save_file_name):
            
            data = self.__hit_api_and_get_data(URL_WITH_TICKER)
            
            key = 'Time Series (Daily)' if config.get_property("time_series_type") == "Daily" else f"{config.get_property("time_series_type")} Time Series"
            
            data = data[key]
            df = pd.DataFrame(columns=['Date', 'Low', 'High', 'Close', 'Open', 'Volume'])
            
            for k, v in data.items():
                date = dt.datetime.strptime(k, '%Y-%m-%d')
                data_row = [date.date(), float(v['3. low']), float(v['2. high']),
                            float(v['4. close']), float(v['1. open']), float(v['5. volume'])]
                
                df = pd.concat([df, pd.DataFrame([data_row], columns=df.columns)], ignore_index=True)
            
            self.__save_data_frame(df, save_file_name=save_file_name)
            
        else:
            df = pd.read_csv(save_file_name)
            
        return df

    pass