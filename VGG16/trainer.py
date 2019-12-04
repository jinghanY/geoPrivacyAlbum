# Ayan Chakrabarti <ayanc@ttic.edu>
import tensorflow as tf
import numpy as np

class train:
    def __init__(self,model,labels,lr,mom,wd):
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=model.out))
        self.accuracy1 = tf.nn.in_top_k(model.out,labels,1)
        self.accuracy5 = tf.nn.in_top_k(model.out,labels,5)
        
        if wd > 0.:
            # Define L2 weight-decay on all non-bias vars
            reg = list()
            for k in model.weights.keys():
                wt = model.weights[k]
                if len(wt.get_shape()) > 1:
                    reg.append(tf.nn.l2_loss(wt))
                    self.reg = tf.add_n(reg)

                    # This is our minimization objective
                    self.obj = self.loss + wd*self.reg
        else:
            self.obj = self.loss
            
        # Set up momentum trainer
        self.opt = tf.train.MomentumOptimizer(lr,mom)

        gv1 = self.opt.compute_gradients(self.obj)
        gv2 = list()
        for gv in gv1:
            if len(gv[1].get_shape()) > 1:
                gv2.append(gv)
            else:
                gv2.append((2.*gv[0],gv[1]))
        self.tstep = self.opt.apply_gradients(gv2)
