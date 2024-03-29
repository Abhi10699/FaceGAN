import torch
import torch.nn as nn
import numpy as np

from PIL import Image

device = "cpu"
weights_path = "./weights/face_generator_v2.pth"
IN_CHANNELS = 100

class FaceGenerator(nn.Module):
    def __init__(self, in_channels):
      super(FaceGenerator, self).__init__()
      self.main = nn.Sequential(
      nn.ConvTranspose2d(in_channels, 1024, 4, 2,0, bias=False),
      nn.BatchNorm2d(1024),
      nn.ReLU(True), # [batch_size, 1024, 2, 2]

      nn.ConvTranspose2d(1024, 512, 4, 2,1, bias=False),
      nn.BatchNorm2d(512),
      nn.ReLU(True), # [batch_size, 512, 7, 7]

      nn.ConvTranspose2d(512, 256, 4, 2,1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(True), # [batch_size, 256, 14, 14]

      nn.ConvTranspose2d(256, 128, 4, 2,1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(True), # [batch_size, 256, 28, 28]
      
      nn.ConvTranspose2d(128,3, 4, 2,1, bias=False),
      nn.Sigmoid(), # [batch_size, 1, 32, 32]
    )
    def forward(self, x):
      return self.main(x)
    
    
def load_model():
  model = FaceGenerator(IN_CHANNELS).to(device=device)
  model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)), strict=True)
  model = model.eval()
  
  print("[!] Model Loaded..")
  return model

def generate(model):
  noise = torch.randn((1,IN_CHANNELS, 1, 1)).to(device)
  image_op = model(noise).squeeze()
  image_op = image_op.permute(1,2,0).detach().cpu().numpy()
  image_op = image_op * 255.0
  image_op = image_op.astype(np.uint8)
  image_op = Image.fromarray(image_op)
  image_op = image_op.resize((256, 256),resample=Image.ADAPTIVE)
  return image_op    
  