# For this notebook to run with updated APIs, we need torch 1.12+ and torchvision 0.13+

import os
import pathlib
from PIL import Image
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision import datasets
from torch import nn
from pathlib import Path
from torch.utils.data import Dataset


from torchinfo import summary
device = "cuda" if torch.cuda.is_available() else "cpu"

image_path="C:\\Users\\soula\\Downloads\\archive\\dataset"
image_train=image_path+"/train"
image_test=image_path+"/test"


    
#find class names from a directory
def find_classes(path):
    classes = sorted([entry.name for entry in list(os.scandir(path))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

class ImageFolderCustom(Dataset):
    
    def __init__(self, targ_dir, transform=None):
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)
        
    def load_img(self,index):
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self,index):
        img = self.load_img(index)
        class_name  = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        if self.transform:
            return self.transform(img), class_idx 
        else:
            return img, class_idx
IMG_SIZE = 224


manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])  
train_data_custom = ImageFolderCustom(targ_dir=image_train,transform=manual_transforms)
test_data_custom = ImageFolderCustom(targ_dir=image_test,transform=manual_transforms)

train_dataloader = DataLoader(dataset=train_data_custom, 
                                     batch_size=1,  
                                     num_workers=0, 
                                     shuffle=True) 

test_dataloader = DataLoader(dataset=test_data_custom, 
                                    batch_size=1, 
                                    num_workers=0, 
                                    shuffle=False) 


image_batch, label_batch = next(iter(train_dataloader))
image, label = image_batch[0], label_batch[0]
print(f'Image shape before patching : {image.shape}')

class LPoFP(nn.Module):
    def __init__(self, in_channels=3,patch_size=16, embedding_dim=768):
        super().__init__()
        self.num_patches = (IMG_SIZE * IMG_SIZE) // patch_size**2
        
        self.CreatePatch= nn.Conv2d(in_channels=in_channels,
                           out_channels=embedding_dim, 
                           kernel_size=patch_size,
                           stride=patch_size,
                           padding=0)  ## Note : Conv2d has autograd
        
        
        self.flatten= nn.Flatten(start_dim=2, end_dim=3)## Note : Flatten has autograd
        
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim), # [batch_size, number_of_tokens, embedding_dimension]
                          requires_grad=True) 
        
        self.position_encoding=nn.Parameter(torch.randn(1,self.num_patches+1,embedding_dim),
                                            requires_grad=True)
        
    def forward(self,x):
        x=self.CreatePatch(x)
        x=self.flatten(x)
        x=x.permute(0,2,1)
        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)
        x=torch.cat((class_token, x), dim=1)
        x=x + self.position_encoding
        return x

class MultiHeadAttention(nn.Module):
    
    def __init__(self,embedding_dim=768,num_heads=12,dropout=0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=dropout,
                                                    batch_first=True)
    def forward(self,x):
        x=self.layer_norm(x)
        attn_output, _  =self.multihead_attn(query=x, key=x, value=x, need_weights=False)
        
        
        return attn_output


class MLP(nn.Module):
    def __init__(self,embedding_dim=768,mlp_size=3072,dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp=nn.Sequential(
               nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
               nn.GELU(),
               nn.Dropout(p=dropout),
               nn.Linear(in_features=mlp_size, out_features=embedding_dim),
               nn.Dropout(p=dropout)
               
               )

    def forward(self,x):
        x=self.layer_norm(x)
        x=self.mlp(x)
        
        
        return x

class TransformerBlock(nn.Module):
    def __init__(self,embedding_dim=768,
                 num_heads=12,
                 mlp_size=3072,
                 dropout_mlpBlock=0.1
                 ):
        super().__init__()
        self.MultiHeadSelfAttention = MultiHeadAttention(embedding_dim=embedding_dim,
                                                        num_heads=num_heads,
                                                        dropout=0)
        self.MLP = MLP(embedding_dim=embedding_dim, 
                       mlp_size=mlp_size, 
                       dropout=dropout_mlpBlock)
        
        
    def forward(self,x):
        x=self.MultiHeadSelfAttention(x) + x
        x=self.MLP(x) + x
        return x
        
class ViT(nn.Module):
    def __init__(self,in_channels=3,
                 patch_size=16, 
                 embedding_dim=768, 
                 num_heads=12,
                 mlp_size=3072,
                 dropout_mlpBlock=0.1,
                 num_classes=3,
                 num_transformer_layers=12
                 ):
        super().__init__()
        self.patch_embedding = LPoFP(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
        
        self.transformer_encoder = nn.Sequential(*[TransformerBlock(embedding_dim=embedding_dim,
                                                                    num_heads=num_heads,
                                                                    mlp_size=mlp_size) for _ in range(num_transformer_layers)])
                                                   
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, 
                      out_features=num_classes))                                

    def forward(self,x):
        x=self.patch_embedding(x)
        x=self.transformer_encoder(x)
        x = self.classifier(x[:, 0])
        return x
    
    
vit = ViT(num_classes=3)
print(summary(model=vit, 
         input_size=(1, 3, 224, 224),device="cuda"))




