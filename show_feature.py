from PIL import Image
import numpy as np


hidden_size = 100
image_size = 28*28
layer1_weight = np.load(r"./para/layer1_weight_"+str(hidden_size)+".npy")
layer2_weight = np.load(r"./para/layer2_weight_"+str(hidden_size)+".npy")

'''''
---normoliaze function---
turn the input to positive number;and normalize to 0-1
'''''
def normolized(input):
    min = np.min(input)
    input = input - min
    max = np.max(input)
    input = input/max
    return input

layer1_weight = layer1_weight.transpose()

print layer1_weight.shape
print layer2_weight.shape

'''''
save the pic pi that maximize the output of hidden_unit hi
pij = Wij/sum(Wij**2) for j from 0 to input size
'''''
for w,i in zip(layer1_weight, range(layer1_weight.__len__())):
    sqrt_w = w**2
    sum_sqrt_w = np.sum(sqrt_w)**0.5
    feature = []
    for j in w:
        xj = j/sum_sqrt_w
        feature.append(xj)
    feature = normolized(feature)
    feature = np.array(feature)*255
    feature = feature.reshape(28,28)
    feature_pic = Image.fromarray(np.uint8(feature))
    feature_pic.save("./feature/feature_1_"+str(i)+".png")