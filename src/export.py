import torch
import torch.onnx
from cnn import Net


#TODO Get this working
# https://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html
# Online web service that enhances photography. 

# A model class instance (class not shown)
model = Net()
model.train(False)

# Load the weights from a file (.pth usually)
state_dict = torch.load("model.pth")

# Load the weights now into a model net architecture defined by our class
print ("loading model")
model.load_state_dict(state_dict['model_state_dict'])  


# Input to the model
x = torch.randn(10, 3, 224, 224, requires_grad=True)

# Export the model
torch_out = torch.onnx._export(model,               # model being run
                               x,                   # model input (or a tuple for multiple inputs)
                               "model.onnx",        # where to save the model (can be a file or file-like object)
                               export_params=True)  # store the trained parameter weights inside the model file
print ("exported model")
