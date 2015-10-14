# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from nlpNet import word_vec, conv_layer, pool_layer, layer, hinge_loss

def word_embeddings_demo():
    l1 = word_vec(word_dim=2)
    l2 = conv_layer(word_dim=2, K=(4,0), lrate=1.0, c=20)
    l3 = pool_layer()
    l4 = layer(dim=(4,1),lrate=1.0,f=lambda x: x,df=lambda x: 1,
               drp_rate=0.0, c=30)
    l5 = layer(dim=(1,1),lrate=1.0,f=np.tanh, df=lambda x: (1-np.square(x)),
               drp_rate=0.5, c=30)
               
    h = hinge_loss(m=1)
    
    A = ['a1','a2','a3', 'a4', 'a5', 'a6', 'a7','ab1','ab2', 'ab3','ab4','ab5', 'ab6']
    B = ['b1','b1','b2','b3', 'b4', 'b5', 'b6', 'b7', 'ab1','ab2','ab3','ab4','ab5', 'ab6']
    C = ['c1','c2','c3', 'c4', 'c5', 'c6', 'c7']
    
    def train(words1, words2):                
        for _ in range(5):
            for p_word in words1:
                for n_word in words2:
                    
                    #Forward pass
                    z1_p, p_word = l1.forward(p_word)
                    z2_p = l2.forward(z1_p)
                    z3_p, z3_ind_max_p = l3.forward(z2_p)
                    z4_p, v4_p, drp4_p = l4.forward(z3_p)
                    z5_p, v5_p, drp5_p = l5.forward(z4_p)
                    
                    z1_n, n_word = l1.forward(n_word)
                    z2_n = l2.forward(z1_n)
                    z3_n, z3_ind_max_n = l3.forward(z2_n)
                    z4_n, v4_n, drp4_n = l4.forward(z3_n)
                    z5_n, v5_n, drp5_n = l5.forward(z4_n)
        
                    #Hinge loss
                    Z = h.forward(z4_p[0], z4_n[0])
                    dE_dZ = h.backward(Z)
                    
                    #Backward deltas
                    d1_p = l5.backward([dE_dZ[0]], z5_p, drp5_p)
                    d2_p = l4.backward([dE_dZ[0]], z4_p, drp4_p)
                    d3_p = l3.backward(d2_p, z3_ind_max_p, z2_p)
                    d4_p = l2.backward(d3_p,z1_p)
                    
                    d1_n = l5.backward([dE_dZ[1]], z5_n, drp5_n)
                    d2_n = l4.backward([dE_dZ[1]], z4_n, drp4_n)
                    d3_n = l3.backward(d2_n, z3_ind_max_n, z2_n)
                    d4_n = l2.backward(d3_n,z1_n)
                    
                    #Update learned parameters
                    l5.update([dE_dZ[0]], z5_p, v5_p)
                    l4.update([dE_dZ[0]], z4_p, v4_p)
                    l2.update(d3_p, z1_p)
                    l1.update(d4_p, p_word)
                    
                    l5.update([dE_dZ[1]], z5_n, v5_n)
                    l4.update([dE_dZ[1]], z4_n, v4_n)
                    l2.update(d3_n, z1_n)
                    l1.update(d4_n, n_word)
                    
    def plot_word_vecs(words, color):
        Px, Py = [l1.W[word][0] for word in words], [l1.W[word][1] for word in words]
        plt.scatter(Px,Py, color=color)

    #Plot word embedding before trainning.
    for word in A+B+C:
        if word in l1.W: continue
        l1.W[word] =  np.random.uniform(0,1.0, 2)

    plot_word_vecs(A, 'blue')
    plot_word_vecs(B, 'red')
    plot_word_vecs(set(A).intersection(B), 'yellow')
    plot_word_vecs(C, 'green')

    plt.show()
    plt.close()
            
    train(A, B)
    train(A, C)
    train(B, C)
    
    #Plot word embeddings after training.
    plot_word_vecs(A, 'blue')
    plot_word_vecs(B, 'red')
    plot_word_vecs(set(A).intersection(B), 'yellow')
    plot_word_vecs(C, 'green')

    plt.show()
    plt.close()

    
if __name__ == '__main__':
    word_embeddings_demo()
    
