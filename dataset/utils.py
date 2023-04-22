import torch
import matplotlib.pyplot as plt

def show_img(img,  resize = True, title = None):
  '''
  @params:
    img: shape of torch.Size([3, 256, 256]) for ex, torch tensor

  '''

  img = (img+1)/2
  # clip at 1 for warnings
  #img = img.where( img <= torch.tensor(1.0) , torch.tensor(1.0) ) # set to 1 if false
  # clip 0s
  #img = img.where( img >= torch.tensor(0.0) , torch.tensor(0.0) ) # set to 1 if false

  if resize:
    img =  torch.nn.functional.interpolate((img).unsqueeze(0),(1024,2048),mode = "bicubic")

  # looks like it doesnt work tho 
  #img = img.where( img <= torch.tensor(255.0) , torch.tensor(255.0) ) # set to 1 if false
  # clip 0s
  #img = img.where( img >= torch.tensor(0.0) , torch.tensor(0.0) ) # set to 1 if false
  
  plt.figure(figsize = (16,8))
  plt.title(title)
  plt.imshow( img[0].permute(1, 2, 0) )# tensor_image.permute(1, 2, 0)  )
