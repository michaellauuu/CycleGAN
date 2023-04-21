
def set_requires_grad(model, requires_grad = False):
  '''set require grad for the model to True/False'''
  for param in model.parameters():    
    param.requires_grad = requires_grad