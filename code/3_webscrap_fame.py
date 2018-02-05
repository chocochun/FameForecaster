from bs4 import BeautifulSoup
import requests
from lxml import html
import pandas as pd
#import wikipedia
import xmltodict 
import requests
import re  
import pickle
import urllib

# python3
#from urllib.request import urlopen


class Artist():
    '''
    A class to webscrap the artist's fame information from whoisbigger
    '''
    def __init__(self):
        self.name = []
        cleandata = pickle.load(open("/Users/minchunzhou/Desktop/insight/clean_data.pickle", "rb"))
        self.allartist = cleandata.artist.unique()
        self.allartist[ self.allartist == "Paul Cezanne" ] = "Paul Cézanne"
                
        self.ini_name = ['Significance','Celebrity', 'Gravitas', 'RanksAvail', 'isPerson', 'Fame', 
                    'hasBirthInfo', 'birthYear', 'deathYear', 'SignificanceRank', 'CelebrityRank', 
                    'GravitasRank', 'FameRank', 'SignificanceRankPercentile', 
                    'CelebrityRankPercentile', 'GravitasRankPercentile', 'FameRankPercentile', 
                    'NormalPagerankRank', 'PersonPagerankRank']
        
        self.ini_name_all = ["all_" + s  for s in self.ini_name]
        
        for i in range(0, len(self.ini_name)):
            setattr(self, self.ini_name_all[i], [])                

        self.init_fame_single()

    def init_fame_single(self):
        
        for i in range(0, len(self.ini_name)):
            setattr(self, self.ini_name[i], 0)                

        
    def number_of_google_result(self):
        '''Return the number of google search result'''
        
        # get google page
        link = "https://www.google.com/search?q=" + self.name
        page  = requests.get(link)
        soup = BeautifulSoup(page.text, 'html.parser')

        # find the result line
        searchresult = soup.find(id="resultStats")

        # get the text
        searchresult_text = searchresult.get_text()

        # get numbers only
        self.google_search = self.retun_number_only(searchresult_text)

        #print("Number of Google search result: " + str(self.google_search))

        
    def length_of_wiki(self):
        ''' Return the length of wikipedia page '''
        
        try: 
            testpage = wikipedia.page(self.name)
            self.wiki_length = len(testpage.content)
        except:
            # print self.name + " Wiki error, return Empty"
            self.wiki_length = 0

        # print("Length of Wiki page: " + str(self.wiki_length))

    #@classmethod
    def get_fame(self):
        ''' Get fame information from http://www.whoisbigger.com/ '''
        
        # get whoisbigger page
        url = "http://www.whoisbigger.com/download_entity.php?entity=entity-" + self.name.lower().replace(" ","-")

        # open page
        u = urllib.urlopen(url)
        #u = urlopen(url)
        try:
            html = u.read()
        finally:
            u.close()

        # if this person is famous
        if html != 'Error downloading this file.':
            
            try:
            
                dat = html.split(",\"")
                startnumber = 16

                # if the search is a person
                if dat[7].replace('"', '').split(":")[1] == "1":
                    startnumber += 6

                for i in range(3,startnumber):
                    fameinfo = dat[i].replace('"', '').split(":")  

                    # dynamic create variable
                    setattr(self, fameinfo[0], fameinfo[1])
                    
            except Exception:
                self.init_fame_single()
        else:
            self.init_fame_single()
        
                #print(fameinfo)
                    

    @classmethod
    # Function to return numbers only
    def retun_number_only(self, textneed):
        rx = re.compile('[^0-9.]')
        s = rx.sub('', textneed)
        return(float(s))
    
    
    def loop_all_artist(self):
        
        
        self.all_number_of_google_result = []

        for i in range(0, len(self.allartist)):
            
            self.name = self.allartist[i]
                        
            self.get_fame()

            #self.all_number_of_google_result.append(self.number_of_google_result())
            
            #self.length_of_wiki()
            #self.number_of_google_result()
            if (i % 100 == 0):

                print("Finish on "+ str(i) + "th Artist: " + self.name + "/n Significance: " + str(self.Significance))

            for i in range(0, len(self.ini_name)):
                getattr(self, self.ini_name_all[i] ).append(getattr(self, self.ini_name[i] ))
              
    def combine_dataframe(self):
        
        self.result = pd.DataFrame()
        self.result['artist'] = self.allartist
        #self.result['google_result'] = self.all_number_of_google_result

        
        for i in range(0, len(self.ini_name)):
            self.i = i
            self.result[self.ini_name[i]] = getattr(self, self.ini_name_all[i])
            
    
    def save_feature(self):
        
        pickle.dump(self.result,open("/Users/minchunzhou/Desktop/insight/famefeature.pickle", "wb"))
        
            
    def run(self):  
        self.loop_all_artist()
        self.combine_dataframe()
        self.save_feature()

if __name__ == '__main__':
    
    Artist = Artist()
    Artist.run()


