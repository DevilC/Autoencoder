import theano
import theano.tensor as T
import numpy as np
import Load_Data as data
from PIL import Image
import os

''''
init weight function
W_bound--type: float -- the range bound of weight
shape -- type: 2d array -- the shape of weight matrix
'''''
def weight_init(W_bound, shape):
    w = np.random.uniform(-W_bound, W_bound, shape)
    return w

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

'''''
---load the weight of encoder from file---
'''''
def para_load(encoder):
    try:
        file = open(r"./para/layer1_weight_"+str(hidden_size)+".npy")
    except IOError:
        print "weight file don't exit!"
    if os.path.exists(r"./para/layer1_weight_"+str(hidden_size)+".npy"):
        temp = np.load(r"./para/layer1_weight_"+str(hidden_size)+".npy")
        encoder.layer1.W.set_value(temp)
        temp = np.load(r"./para/layer1_b_"+str(hidden_size)+".npy")
        encoder.layer1.B.set_value(temp)
        temp = np.load(r"./para/layer2_weight_"+str(hidden_size)+".npy")
        encoder.layer2.W.set_value(temp)
        temp = np.load(r"./para/layer2_b_"+str(hidden_size)+".npy")
        encoder.layer2.B.set_value(temp)

'''''
---the class of layer---
-input{type: theano tensor; as the input of this layer}
-input_num{type: int; the number of input}
-out_num{type: int; the number of output}
class variable:
    self.W{type: shared Variable; the weight connect from input and layer unit}
    self.B{type: shared Variable; the bias connect from input and layer unit}
    self.output{type: tensor; the output of layer,sigmod function}
    self.para{type: list; the parament of layer}
'''''
class layer():
    def __init__(self, input, input_num, out_num):
        self.input_num = input_num
        self.output_num = out_num

        W_bound = np.sqrt(out_num/(input_num + hidden_size))
        self.W = theano.shared(weight_init(W_bound, (input_num,out_num)))

        b_value = weight_init(0, (1,out_num))
        self.B = theano.shared(b_value)

        self.y = T.dot(input, self.W) + self.B
        self.output = 1/(1+T.exp(-self.y))

        self.para = [self.W, self.B]

input_size = 28*28  #size of input
hidden_size = 100   #size of hidden layer
output_size = 28*28 #size of output layer
sample_num = 50000  #number of sample
step = 0.0005   #step

'''''
---the class of encoder---
a three layers, full connected network;
class variable:
self.layer1{type: layer; the first layer of encoder(hidden layer)
self.layer2{type: layer; the second layer of encoder(output layer)
self.para{type: list; all of encoder's para
self.cost{type: tensor; the cost of model
self.train{type: theano.function; the train function
self.coder{type: theano.function; the coder function
'''''
class encoder:
    def __init__(self):
        l1_i = T.vector('input')
        l1_o = T.vector('layer1_output')
        l2_i = T.vector('layer2_input')
        l2_o = T.vector('layer2_output')

        self.layer1 = layer(l1_i, input_size, hidden_size)
        l1_o = self.layer1.output

        l2_i = l1_o
        self.layer2 = layer(l2_i, hidden_size, output_size)
        l2_o = self.layer2.output

        self.para = self.layer1.para+self.layer2.para

        self.cost = T.sum((l2_o-l1_i)**2)

        self.grads = T.grad(self.cost, self.para)
        update =[]
        for para_i, grad_i in zip(self.para, self.grads):
            update.append((para_i, para_i - (step*grad_i)))

        self.train = theano.function([l1_i], self.cost, updates = update)
        self.coder = theano.function([l1_i], self.layer2.output)

my_encoder = encoder()
para_load(my_encoder)
for p in range(100):
    code = my_encoder.coder(data.train_setX[p])[0]*255
    code = np.array(code)
    code = code.reshape(28,28)
    pic = Image.fromarray(np.uint8(code))
    pic.save("./pic/"+str(p)+"_c.png")

    pic = np.array(data.train_setX[p] * 225)
    pic = pic.reshape(28, 28)
    pic = Image.fromarray(np.uint8(pic))
    pic.save("./pic/"+str(p)+".png")

'''''
train process:
'''''
'''''
print "training!"

for j in range(10):
    for s in range(30):
        print str(j)+'-'+str(s)
        all_cost = 0
        for x in data.train_setX:
            all_cost += my_encoder.train(x)
        print all_cost

    np.save(r"./para/layer1_weight_"+str(hidden_size)+".npy", my_encoder.layer1.W.get_value())
    np.save(r"./para/layer1_b_"+str(hidden_size)+".npy", my_encoder.layer1.B.get_value())
    np.save(r"./para/layer2_weight_"+str(hidden_size)+".npy", my_encoder.layer2.W.get_value())
    np.save(r"./para/layer2_b_"+str(hidden_size)+".npy", my_encoder.layer2.B.get_value())
'''''