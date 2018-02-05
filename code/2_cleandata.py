# clean data
import pandas as pd
import numpy as np
import pickle
import re
from datetime import datetime


class Clean_rawdata():
    '''
    A class to clean raw data from blouinartsales
    '''
    
    def __init__(self):
        
        # load data
        self.tempdata = pickle.load(open("/Users/minchunzhou/Desktop/insight/data/10000.pickle", "rb"))
        
    def get_artinfo(self):

        # get year and workname
        df = self.tempdata.name.str.rsplit(',',1, expand=True)
        df.columns = ['workname','year']
        df = df.replace('\n','', regex=True)
        df = df.replace('Circa','', regex=True)
        df = df.replace('Printed','', regex=True)

        df.year = df.year.str.lstrip()

        df.year[(df.year != "" ) &  ( df.year.notnull()) ]  = [x[:4] for x in df.year[(df.year != "" ) &  ( df.year.notnull()) ] ] 
        df.columns = ['workname','workyear']

        self.df_year = df
        
    def get_price_currency(self):
        
        # get price
        df = self.tempdata.price.str.split(' ', expand=True)
        df = df.replace('\n','', regex=True)
        df.columns = [ "sale_sign","sale_in_USD", "USD", "sale_in_EUR", "EUR", "sale_in_GBP", "GBP", "other" ]
        # wrong index shift
        pre = ['GBP']
        df[df['other'].isin(pre)] = df[df['other'].isin(pre)].shift(-1,axis=1)

        df = df.drop(['sale_sign','USD','EUR','GBP','other'], axis=1)

        df.sale_in_USD = df.sale_in_USD.str.replace(",","")
        df.sale_in_EUR = df.sale_in_EUR.str.replace(",","")
        df.sale_in_GBP = df.sale_in_GBP.str.replace(",","")

        self.df_price = df
        
    def get_price_sale(self):
        
        # sale price
        # hammer or premium
        sale = self.tempdata.sale.str.replace('\n','')
        df = sale.str.lstrip(' ').str.split(' ', expand=True)
        df.columns = ['sign','orig_price','orig_currency','premium','other']
        pre = ['Hammer', 'Premium']

        df[df['orig_currency'].isin(pre)] = df[df['orig_currency'].isin(pre)].shift(1,axis=1)

        df = df.drop(['sign','other'], axis=1)
        df_sale = df
        df_sale.orig_price = df_sale.orig_price.str.replace(",","")
        df_sale.orig_price = pd.to_numeric(df_sale.orig_price)
        self.df_sale = df_sale
        
    def get_estimate(self):

        # estimate
        estimate = self.tempdata.estimate.str.replace('\n','')

        df1 = estimate.str.rsplit(' ',1, expand=True)
        df1.columns = ['estimate','estimate_currency']

        df2 = df1.estimate.str.split('-', expand=True)
        df2.columns = ['estimate_low','estimate_up']

        df = pd.concat([df1, df2], axis=1)

        trim = re.compile(r'[^\d.,]+')

        for num in df.index:
            if pd.notnull(df.estimate_up[num]):
                df.estimate_up[num] =  trim.sub('',df.estimate_up[num])

            if pd.notnull(df.estimate_low[num]):
                df.estimate_low[num] =  trim.sub('',df.estimate_low[num])

        df.estimate_low = df.estimate_low.str.replace(",","")
        df.estimate_up = df.estimate_up.str.replace(",","")

        df_estimate = df.drop(['estimate'], axis=1)
        df_estimate.estimate_up = pd.to_numeric(df_estimate.estimate_up)
        df_estimate.estimate_low = pd.to_numeric(df_estimate.estimate_low)
        
        self.df_estimate = df_estimate
        
    def get_auction_date(self):
        
        auction_date = self.tempdata.auction_date.str.replace('\n','')

        df = auction_date.str.rsplit(',',1, expand=True)
        df.columns = [ 'auction_date', 'auction_year' ]

        df = df.drop(['auction_date'],axis=1)

        self.auction_date = df


    def get_auction_location(self):
        
        df = self.tempdata.location.str.rsplit(',',1, expand=True)
        df.columns = ['auction_house','location']
        self.df_auction_location = df

    def get_artist_bio(self):
        
        # bio
        bio = self.tempdata.bio.str.replace('\n','')
        bio = bio.str.replace('(','')
        bio = bio.str.replace(')','')

        df = bio.str.rsplit(',',1, expand=True)
        df.columns = ['artist_country','artist_year']

        df1 = df.artist_year.str.rsplit('-',1, expand=True)
        df1.columns = ['artist_birth_year','artist_death_year']

        self.df_bio = df

    def get_art_medium(self):
        
        df_medium = self.tempdata.medium.str.replace('\n','')
        self.df_medium = pd.DataFrame(df_medium)

        
    def get_art_measurement(self):

        # measurement
        measurement  = self.tempdata.measurement.str.replace('\n','')
        df = measurement.str.split('by', expand=True)
        df.columns = ['height','width','depth']
        df  = df.replace('\(height\)','', regex=True)
        df  = df.replace('\(width\)','', regex=True)
        df  = df.replace('\(depth\)','', regex=True)

        # height
        self.df_height = df.height.str.split('(', expand=True)
        self.df_height.columns = ['height_inch','height_cm']
        self.df_height.height_inch = self.df_height.height_inch.str.replace('in.','')
        self.df_height.height_cm = self.df_height.height_cm.str.replace('cm.\)','')

        # width
        self.df_width = df.width.str.split('(', expand=True)
        self.df_width.columns = ['width_inch','width_cm']
        self.df_width.width_inch = self.df_width.width_inch.str.replace('in.','')
        self.df_width.width_cm = self.df_width.width_cm.str.replace('cm.\)','')

        # depth
        self.df_depth = df.depth.str.split('(', expand=True)
        self.df_depth.columns = ['depth_inch','depth_cm']
        self.df_depth.depth_inch = self.df_depth.depth_inch.str.replace('in.','')
        self.df_depth.depth_cm = self.df_depth.depth_cm.str.replace('cm.\)','')

        self.df_height.height_inch = self.df_height.height_inch.convert_objects(convert_numeric=True)
        self.df_width.width_inch = self.df_width.width_inch.convert_objects(convert_numeric=True)


    def run(self):  
        
        self.get_artinfo()
        self.get_price_currency()
        self.get_price_sale()
        self.get_estimate()
        self.get_auction_date()
        self.get_auction_location()
        self.get_artist_bio()
        self.get_art_medium()
        self.get_art_measurement()
        
        self.df_clean = pd.concat([self.tempdata[['picurl','weburl', 'num_author']],
                              self.df_height, self.df_width, self.df_depth, self.df_medium, 
                              self.df_bio,self.df_auction_location,self.auction_date,
                              self.df_estimate,self.df_sale,self.df_price, self.df_year,
                             need], axis=1)
        
        pickle.dump(self.df_clean,open("/Users/minchunzhou/Desktop/insight/clean_data.pickle", "wb"))

if __name__ == '__main__':
    
    Clean_rawdata = Clean_rawdata()
    Clean_rawdata.run()    

    

    
