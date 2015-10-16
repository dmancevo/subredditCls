# -*- coding: utf-8 -*-

############################################################
#The code below was borrowed (and slightly modified) from: #
#https://github.com/dmancevo/neural_nets                   #
############################################################

import numpy as np

class softmax_layer(object):
    '''
    Softmas layer for multiclass classification.
    '''
    
    def __init__(self):
        
        self.Kdelta = lambda i, k: 1.0 if i==k else 0.0
    
    def forward(self, V):

        return np.exp(V)/np.sum(np.exp(V))
    
    def backward(self, dE_dZ, Z, V):
        
        dE_dV = []
        for j in range(len(V)):

            dE_dV.append(np.sum([dE_dZ[i] * Z[i]\
            * (self.Kdelta(i,j) - Z[j])\
            for i in range(len(dE_dZ))]))
            
        return dE_dV

class hinge_loss(object):
    '''
    Hinge loss layer
    (for unsupervised learning of word embeddings).
    '''
    
    def __init__(self, m):
        '''
        m: margin
        '''
        
        self.m = m
    
    def forward(self, x1, x2):
        
        return np.max(np.array([0, (self.m - x1 + x2)]))
    
    def backward(self, Z):
        
        if Z > 0:
            return np.array([-1, 1])
        else:
            return np.zeros(2)
    
class layer(object):
    '''
    Standard layer with drop out and max-norm regularization.
    '''
    
    def __init__(self, dim, lrate, f, df, drp_rate, c):
        '''
        dim: dimensions tuple (input, outpu)
        lrate: learning rate
        f: activation function (e.g. tanh, linear)
        df: derivative of activation function
        drp_rate: drop-out rate
        c: max-norm regularization hyperparameter
        '''
        
        self.dim = dim
    
        self.b = [np.random.normal(0,0.01) for i in range(dim[1])]
        self.W = [np.random.normal(0,0.01, dim[0]) for i in range(dim[1])]
        
        self.f = f
        self.df = df
        
        self.lrate = lrate
        self.t = 1
        
        self.drp_rate = drp_rate
        
        self.c = c
    
    def forward(self, V, train=True):
        
        if train:
            drp = np.random.binomial(1,1.0-self.drp_rate,len(V))
            V = V * drp
        else:
            V = V * (1.0-self.drp_rate)
            
        Z = np.array([self.f(self.W[i].dot(V) + self.b[i])\
        for i in range(self.dim[1])])
           
        if train:
            return Z, V, drp
        else:
            return Z
    
    def backward(self, dE_dZ, Z, drp):
        
        dE_dV = []
        
        for j in range(self.dim[0]):
            
            if drp[j] == 0.0:
                dE_dV.append(0.0)
            else:
                dE_dV.append(np.sum([dE_dZ[i] * self.df(Z[i]) * self.W[i][j]\
                for i in range(self.dim[1])]))
            
        return np.array(dE_dV)
            
    
    def update(self, dE_dZ, Z, V):
        
        for i in range(self.dim[1]):
            
            #Update weights
            gW = dE_dZ[i] * self.df(Z[i]) * V
            self.W[i] -= self.lrate * gW / np.sqrt(self.t)
            
            #Max-norm regularization
            W2 = np.linalg.norm(self.W[i],2)
            if W2 > self.c:
                self.W[i] = self.c * self.W[i] / W2
            
            #Update bias
            gb = dE_dZ[i] * self.df(Z[i])
            self.b[i] -= self.lrate * gb / np.sqrt(self.t)
            
            self.t += 1
            
            
class pool_layer(object):
    '''
    Pool layer.
    '''
    
    def __init_(self):
        
        pass
        
    def forward(self, V):
        
        Z, Z_max_ind = [], []
        for v in V:
            v_max_ind = v.argmax()
            Z.append(v[v_max_ind])
            Z_max_ind.append(v_max_ind)
            
        return np.array(Z), Z_max_ind
    
    def backward(self, dE_dZ, Z_max_ind, V):
        
        dE_dV = []
        
        for i in range(len(V)):
            v = V[i]
            dE_dv = np.zeros(len(v))
            dE_dv[Z_max_ind[i]] = dE_dZ[i]
            dE_dV.append(dE_dv)
            
        return dE_dV

class conv_layer(object):
    '''
    Convolution layer
    '''
    
    def __init__(self, word_dim, K, lrate, c):
        
        self.word_dim = word_dim        
        self.W = []
        self.b = []
        
        self.lrate = lrate
        self.t = 1.0
        
        self.c = c
        
        for i in range(K[0]):
            self.W.append(np.random.normal(-0.001,0.001,word_dim))
            self.b.append(np.random.normal(-0.001,0.001))
            
        for i in range(K[1]):
            self.W.append(np.random.normal(-0.001,0.001,2*word_dim))
            self.b.append(np.random.normal(-0.001,0.001))
            
    def forward(self, V):
        
        Z = []
        
        for j in range(len(self.W)):
            z = []
            k = len(self.W[j])
            i = 0
            while i+k <= len(V):
                z.append(self.W[j].dot(V[i:(i+k)])+self.b[j])
                i += self.word_dim
                
            if not z: z.append(0.0)
                
            Z.append(np.array(z))
            
        return Z
    
    def backward(self, dE_dZ, V):
        
        dE_dV = np.zeros(len(V))
        
        for j in range(len(self.W)):
            w = self.W[j]
            k = len(w)
            i, l = 0, 0
            while i+k <= len(V):
                dE_dV[i:(i+k)] += dE_dZ[j][l] * w
                i += self.word_dim
                l += 1
        
        return dE_dV
    
    def update(self, dE_dZ, V):
        
        for j in range(len(self.W)):
            dw = np.zeros(len(self.W[j]))
            db = 0.0
            k = len(dw)
            i, l = 0, 0
            while i+k <= len(V):
                dw += dE_dZ[j][l] * V[i:(i+k)]
                db += dE_dZ[j][l]
                i += self.word_dim
                l += 1
                
            rate = 1/(np.sqrt(self.t))
            self.W[j] -= rate * dw
            self.b[j] -= rate * db
            
            self.t += 1.0
            
            #Max-norm regularization
            W2 = np.linalg.norm(self.W[j],2)
            if W2 > self.c:
                self.W[j] = self.c * self.W[j] / W2
            

    
class word_vec(object):
    '''
    Word vectorizer.
    '''
    
    def __init__(self, word_dim):
        
        self.word_dim = word_dim
        self.W = {}
        self.G2 = {}
                
    def forward(self, sentence):
        
        Z = np.array([])
        new_sentence = []
                    
        for word in sentence.split():
            
            word = word.lower()
            new_sentence.append(word)
            
            if word not in self.W:
                self.W[word] = np.random.uniform(0,1.0, self.word_dim)
                
            if word not in self.G2:
                self.G2[word] = 0.0
                
            Z = np.hstack((Z,self.W[word]))
                        
        return Z, ' '.join(new_sentence)
        
    
    def update(self, dE_dZ, sentence):
        
        i = 0
        for word in sentence.split():
            
            rate = 1/(1+np.sqrt(self.G2[word]))
            self.G2[word] += np.square(dE_dZ[i:(i+self.word_dim)])
            
            self.W[word] -= rate * dE_dZ[i:(i+self.word_dim)]
            
            i += self.word_dim
        
