# loading package
from bs4 import BeautifulSoup
import requests
from lxml import html
import pandas as pd
import pickle
import time


class Scrap_blouinart(): 
    
    def __init__(self):
        
        # init page 
        self.link = "https://www.blouinartsalesindex.com/auctions/--"
        
        # login info
        self.loginuser = {"j_username": "username", "j_password": "password"}
        self.login_page = "https://www.blouinartsalesindex.com/j_spring_security_check"
        self.sess = requests.session()
        self.resp = self.sess.post(self.login_page, self.loginuser)

        # scrap_info
        self.ini_name = ['num_author', 'picurl', 'weblink' ,'price', 'sale', 'estimate','auction_date', 
                        'location', 'auction_name' , 'artist', 'bio', 'name', 'medium', 
                        'measurement',]
        
        self.ini_name_all = ["all_" + s  for s in self.ini_name]
        

    def init_fame_chunk(self):
        
        for i in range(0, len(self.ini_name)):
            setattr(self, self.ini_name_all[i], [])                
        
    def init_fame_single(self):
        
        for i in range(0, len(self.ini_name)):
            setattr(self, self.ini_name[i], [])                
        
    def scrap(self):
                
        try:
            self.sess = requests.session()
            self.resp = self.sess.post(self.login_page, self.loginuser)

            result = self.sess.get(self.weblink)
            soup = BeautifulSoup(result.text, 'html.parser')

            mydivs = soup.find('h2', class_='mt-10')
            workinfo = mydivs.find_all('p')

            NumAuthor = len(mydivs.find_all("span",class_="artworkyear"))
            self.num_author = NumAuthor

            # picurl 
            if (soup.find('img',id="artworkIndex_10") == None):
                self.picurl = soup.find('img',id="artworkIndex_0").get("src")

            else:
                self.picurl = soup.find('img',id="artworkIndex_10").get("src")

            # price
            self.price = mydivs.find_all('p')[2*(NumAuthor-1) +2].text


            # sale 
            self.sale = mydivs.find_all('p')[2*(NumAuthor-1) +4].text

            # estimate
            self.estimate = mydivs.find_all('div')[2*(NumAuthor-1) + 5].text

            # auction date
            self.auction_date = mydivs.find_all('div')[2*(NumAuthor-1) + 8].text

            # auction location and house
            self.location = mydivs.find_all('p')[2*(NumAuthor-1) +7].text
            self.auction_name = mydivs.find_all('p')[2*(NumAuthor-1) +8].text

            # artist 
            if (NumAuthor > 1):
                self.artist = mydivs.find_all('a')[0].text + " and "  + mydivs.find_all('a')[3].text

                # bio
                self.bio = mydivs.find_all('p')[0].text + " and " + mydivs.find_all('p')[2].text

            else:
                self.artist = mydivs.find_all('a')[0].text 

                # bio
                self.bio = mydivs.find_all('p')[0].text

            # work name
            self.name = mydivs.find_all('p')[1].text

            # detail
            detail = soup.find('div', class_='moredetails')

            # medium
            self.medium = detail.find_all('p')[1].text

            # measurement 
            self.measurement = detail.find_all('p')[3].text

        except Exception:
            pass
        

    def combine_dataframe(self):
        
        self.result = pd.DataFrame()
        
        for i in range(1, len(self.ini_name)):
            self.i = i
            self.result[self.ini_name[i]] = getattr(self, self.ini_name_all[i])
        
        
    def looping_website(self):
        
        print("start scraping")   
        
        self.init_fame_chunk()
        
        for i in range(1 ,100001 ):
        #for i in range(1,60000):
                        
            self.init_fame_single()
            self.weblink = self.link + str(i) + "/"
            self.scrap()
            time.sleep(1)

            print self.price

            for j in range(0, len(self.ini_name)):
                getattr(self, self.ini_name_all[j] ).append(getattr(self, self.ini_name[j] ))

                
            if (i % 100 == 0):
                print("Done " + str(i))   

                self.sess = requests.session()
                self.resp = self.sess.post(self.login_page, self.loginuser)
    
            
            if (i % 10000 == 0):
                
                self.combine_dataframe()
                #self.result.to_csv( "/scratch/kangh1/minchun_other/insight/allwebdata/" + str(i) + ".csv" ,encoding='utf-8',index=None)
                #self.result.to_csv( "/Users/minchunzhou/Desktop/insight/new_scrap/" + str(i) + ".csv" ,encoding='utf-8',index=None)
                pickle.dump( self.result, open( "/scratch/kangh1/minchun_other/insight/scrap_all/data/" + str(i) + ".pickle", "wb" ) , protocol=2)
                self.init_fame_chunk()

                print("Save " + str(i))   

            
if __name__ == '__main__':
    
    Scrap_blouinart = Scrap_blouinart()
    Scrap_blouinart.looping_website()
        

    
