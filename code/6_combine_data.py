import pandas as pd
import numpy as np
import pickle
from collections import Counter


class Combine_alldata():
    '''
    A class to build classifier model
    '''
    def __init__(self):
        
        # load data from different source
        self.cleandata = pickle.load(open("/Users/minchunzhou/Desktop/insight/clean_data.pickle", "rb"))
        self.fame = pickle.load(open("/Users/minchunzhou/Desktop/insight/famefeature.pickle", "rb"))
        self.imag = pickle.load(open("/Users/minchunzhou/Desktop/insight/imagefeature.pickle", "rb"))
        
    def combine_data(self):
        
        # change fame  Paul Cezanne - Paul Cézanne
        self.imag = pd.DataFrame(self.imag)
        mystring = "img_feature_"
        self.imag.columns = [ mystring + str(s) for s in (range(1, self.imag.shape[1]+1))]

        self.currentdata = pd.concat([self.cleandata.reset_index(drop=True), self.imag], axis=1)
        self.currentdata.artist[ self.currentdata.artist == "Paul Cezanne" ] = "Paul Cézanne"

        self.jointable = pd.merge(self.currentdata, self.fame, on= 'artist', how="left" )
        #pickle.dump(self.jointable,open("/Users/minchunzhou/Desktop/insight/temp_final_data.pickle", "wb"))
    
    def subset_data(self):
        
        self.jointable["is_famous"] = (self.jointable.Celebrity != 0) *1
        self.jointable["is_alive_auction"] = pd.isnull(self.jointable.yearOfDeath)


        self.jointable.sale_in_USD = pd.to_numeric(self.jointable.sale_in_USD) 
        self.jointable['log_sale_in_USD'] = np.log10(self.jointable.sale_in_USD)

        self.jointable.workyear = pd.to_numeric(self.jointable.workyear, errors='coerce')
        self.jointable = self.jointable.drop( self.jointable.index[pd.isnull(self.jointable.workyear)])
        self.jointable = self.jointable.drop( self.jointable.index[pd.isnull(self.jointable.img_feature_1)])


        self.jointable.orig_price = pd.to_numeric(self.jointable.orig_price)
        self.jointable["mean_estimate"] =   self.jointable.sale_in_USD / self.jointable.orig_price * (self.jointable.estimate_low + self.jointable.estimate_up)/2 
        self.jointable.mean_estimate = np.round(self.jointable.mean_estimate)


        self.jointable  = self.jointable[self.jointable.sale_in_USD >= 1000 ]
        self.jointable['area_in_inch'] = self.jointable.height_inch * self.jointable.width_inch

        self.jointable = self.jointable[self.jointable.artist != "Andy Warhol"]
        self.jointable = self.jointable[self.jointable.artist != "Pablo Picasso"]
        
        
    def artist_onehot(self):
        
        fame_use = ['Celebrity', 'Gravitas', 'RanksAvail', 'isPerson', 'Fame',
                            'hasBirthInfo', 'SignificanceRankPercentile', 
                            'CelebrityRankPercentile', 'GravitasRankPercentile', 'FameRankPercentile', 
                            'NormalPagerankRank', 'PersonPagerankRank']

        data_artist = self.jointable[['artist', 'country', 'yearOfBirth', 'is_famous', 'is_alive_auction']] # + fame_use + 
        artist_unique = data_artist.drop_duplicates()
        words = artist_unique.country
        most_common_words= [word for word, word_count in Counter(words).most_common(20)]
        #print most_common_words
        most_common_country = most_common_words

        data_artist.country[~data_artist.country.isin(most_common_words)] = "Other_artist_country"
        artist_country_dummy = pd.get_dummies(data_artist.country)
        artist_country_dummy.shape
        self.all_artist = pd.concat([data_artist.drop(["artist", "country"],axis=1) , artist_country_dummy], axis=1)

    def art_onehot(self):
        
        data_art = self.jointable[ [ 'height_inch', 'width_inch', 'medium','workname','workyear',
                               'dominantColor', 'brightness', 'ratioUniqueColors',
                               'thresholdBlackPerc', 'highbrightnessPerc','lowbrightnessPerc',
                              'CornerPer','EdgePer','FaceCount'] ] # + list(self.imag.columns) +["picurl", "weburl",] ]# 

        data_art['area_in_inch'] = data_art.height_inch * data_art.width_inch
        words = data_art.medium
        most_common_words= [word for word, word_count in Counter(words).most_common(10)]
        most_common_medium = most_common_words
        #print most_common_words

        data_art.medium[~data_art.medium.isin(most_common_words)] = "Other_medium"
        medium_dummy = pd.get_dummies(data_art.medium)
        dominantColor_dummy = pd.get_dummies(data_art.dominantColor)

        self.all_art = pd.concat( [data_art.drop(['medium','workname',"dominantColor"], axis=1), #medium_dummy, 
                            dominantColor_dummy ] , axis=1)
        
    def auction_onehot(self):

        data_auction = self.jointable[[  'auction_house','location','auction_year', 
                                  'premium',"estimate_low","estimate_up" ,"orig_price"]]

        words = data_auction.auction_house
        most_common_words= [word for word, word_count in Counter(words).most_common(20)]
        data_auction.auction_house[~data_auction.auction_house.isin(most_common_words)] = "Other_auction_house"
        auction_house_dummy = pd.get_dummies(data_auction.auction_house)


        words = data_auction.location
        most_common_words= [word for word, word_count in Counter(words).most_common(10)]
        data_auction.location[~data_auction.location.isin(most_common_words)] = "Other_auction_location"
        location_dummy = pd.get_dummies(data_auction.location)

        premium_dummy = pd.get_dummies(data_auction.premium)
        data_auction.auction_year = pd.to_numeric(data_auction.auction_year)
        self.all_auction = pd.concat( [data_auction.drop(["location", "auction_house", "premium"],axis=1), location_dummy, auction_house_dummy, premium_dummy] ,axis=1)


    def run(self):  
        self.combine_data()
        self.subset_data()
        self.artist_onehot()
        self.art_onehot()
        self.auction_onehot()

        self.cleandata = pd.concat( [ #self.jointable.sale_in_USD.reset_index(drop=True), #all_auction.reset_index(drop=True), 
                        self.all_art.reset_index(drop=True), self.all_artist.reset_index(drop=True) ] , axis=1)

        self.cleandata['aspect_ratio'] = self.cleandata.width_inch / self.cleandata.height_inch
        
        #pickle.dump(self.cleandata,open("/Users/minchunzhou/Desktop/insight/dataformodel.pickle", "wb"))


if __name__ == '__main__':
    
    Combine_alldata = Combine_alldata()
    Combine_alldata.run()   
    print("Done")
    
