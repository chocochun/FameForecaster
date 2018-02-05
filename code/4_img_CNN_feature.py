import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import pickle
from PIL import Image
import requests
import pandas as pd
import numpy as np



class Feature_from_image():
    '''
    Use pytorch to extract features from image
    '''
    
    def __init__(self):
        cleandata = pickle.load(open("/scratch/kangh1/minchun_other/insight/features/clean_data.pickle", "rb"),encoding='latin1')
        #cleandata = pickle.load(open("/Users/minchunzhou/Desktop/insight/clean_data.pickle", "rb"))
        self.allurl = cleandata.picurl.values
    
    def get_all_feature(self):
        self.allfeature = []
        
        # testsave
        self.save_feature()

        self.build_model()
        
        for i in range(0,len(self.allurl)):
            
            try: 
                img_feature = self.get_vector(self.allurl[i])
                self.allfeature.append(img_feature.numpy())

            except:
                img_feature = [None] * 512
                self.allfeature.append(img_feature)
                
            if (i % 100 == 0):
                self.save_feature()
                
        self.save_feature()

                
    def save_feature(self):
        
        pickle.dump(self.allfeature,open("/scratch/kangh1/minchun_other/insight/features/imagefeature.pickle", "wb"), protocol=2)
        #pickle.dump(self.allfeature,open("/Users/minchunzhou/Desktop/insight/imagefeature.pickle", "wb"))
    
    def build_model(self):

        # Load the pretrained model
        self.model = models.resnet18(pretrained=True)
        # Use the model object to select the desired layer
        self.layer = self.model._modules.get('avgpool')
        # Set model to evaluation mode
        self.model.eval()
        self.scaler = transforms.Scale((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    
    # function to get image feature from url
    def get_vector(self,singlepicurl):
                
        # 1. Load the image with Pillow library
        img = Image.open(requests.get(singlepicurl))
        # 2. Create a PyTorch Variable with the transformed image
        t_img = Variable(self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0))
        # 3. Create a vector of zeros that will hold our feature vector
        #    The 'avgpool' layer has an output size of 512
        my_embedding = torch.zeros(512)
        # 4. Define a function that will copy the output of a layer
        def copy_data(m, i, o):
            my_embedding.copy_(o.data)
        # 5. Attach that function to our selected layer
        h = self.layer.register_forward_hook(copy_data)
        # 6. Run the model on our transformed image
        self.model(t_img)
        # 7. Detach our copy function from the layer
        h.remove()
        # 8. Return the feature vector
        return my_embedding    
                
    def run(self):        
        self.get_all_feature()


if __name__ == '__main__':
    
    Feature_from_image = Feature_from_image()
    Feature_from_image.run()

