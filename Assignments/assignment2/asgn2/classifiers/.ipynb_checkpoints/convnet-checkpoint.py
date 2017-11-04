import numpy as np

from asgn2.layers import *
from asgn2.fast_layers import *
from asgn2.layer_utils import *
from asgn2.bn_layer_utils import *


class ConvNet5Layer(object):
  """
  A five-layer convolutional network with the following architecture:

  [conv - relu - conv - relu - 2x2 max pool] * M - [affine - relu ] * N - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=64, filter_size=3,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    
    hidden_dim1 = (num_filters*H*W)/64
    
    # CONV1 params
    self.params['W1'] = np.random.normal(
        0, scale=weight_scale, size=(num_filters, C, filter_size, filter_size)
    )
    self.params['b1'] = np.zeros(num_filters)
    
    # Spatial BatchNorm 1 params
    self.params['gamma1'] = np.ones(num_filters)
    self.params['beta1'] = np.zeros(num_filters)
    
    # CONV2 params
    self.params['W2'] = np.random.normal(
        0, scale=weight_scale, size=(num_filters, num_filters, filter_size, filter_size)
    )
    self.params['b2'] = np.zeros(num_filters)
    
    # Spatial BatchNorm 2 params
    self.params['gamma2'] = np.ones(num_filters)
    self.params['beta2'] = np.zeros(num_filters)
    
    # CONV3 params
    self.params['W3'] = np.random.normal(
        0, scale=weight_scale, size=(num_filters, num_filters, filter_size, filter_size)
    )
    self.params['b3'] = np.zeros(num_filters)
    
    # Spatial BatchNorm 3 params
    self.params['gamma3'] = np.ones(num_filters)
    self.params['beta3'] = np.zeros(num_filters)
    
    # Affine1 params
    self.params['W4'] = np.random.normal(
        0, scale=weight_scale, size=(hidden_dim1, hidden_dim)
    )
    
    self.params['b4'] = np.zeros(hidden_dim)
    
    # Batchnorm params
    self.params['gamma4'] = np.ones(hidden_dim)
    self.params['beta4'] = np.zeros(hidden_dim)
    
    # Affine2 params
    self.params['W5'] = np.random.normal(
        0, scale=weight_scale, size=(hidden_dim, num_classes)
    )
    self.params['b5'] = np.zeros(num_classes)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    gamma1, beta1 = self.params['gamma1'], self.params['beta1']
    
    W2, b2 = self.params['W2'], self.params['b2']
    gamma2, beta2 = self.params['gamma2'], self.params['beta2']
    
    W3, b3 = self.params['W3'], self.params['b3']
    gamma3, beta3 = self.params['gamma3'], self.params['beta3']
    
    W4, b4 = self.params['W4'], self.params['b4']
    gamma4, beta4 = self.params['gamma4'], self.params['beta4']
    
    W5, b5 = self.params['W5'], self.params['b5']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
    # Batchnorm params
    bn_param = {'mode': 'train'}
    sbn_param = {'mode': 'train'}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    conv1_out, conv1_cache = conv_bn_relu_pool_forward(X, W1, b1, gamma1, beta1, conv_param, pool_param, sbn_param)
    conv2_out, conv2_cache = conv_bn_relu_pool_forward(conv1_out, W2, b2, gamma2, beta2, conv_param, pool_param, sbn_param)
    conv3_out, conv3_cache = conv_bn_relu_pool_forward(conv2_out, W3, b3, gamma3, beta3, conv_param, pool_param, sbn_param)
    af1_out, af1_cache = affine_bn_relu_forward(conv3_out, W4, b4, gamma4, beta4, bn_param)
    af2_out, af2_cache = affine_forward(af1_out, W5, b5)
    
    scores = af2_out
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores, y)
    
    loss += 0.5 * self.reg * np.sum(W1*W1)
    loss += 0.5 * self.reg * np.sum(W2*W2)
    loss += 0.5 * self.reg * np.sum(W3*W3)
    loss += 0.5 * self.reg * np.sum(W4*W4)
    loss += 0.5 * self.reg * np.sum(W5*W5)
    
    dout, dW5, db5 = affine_backward(dout, af2_cache)
    dout, dW4, db4, dgamma4, dbeta4 = affine_batchnorm_relu_backward(dout, af1_cache)
    dout, dW3, db3, dgamma3, dbeta3 = conv_bn_relu_pool_backward(dout, conv3_cache)
    dout, dW2, db2, dgamma2, dbeta2 = conv_bn_relu_pool_backward(dout, conv2_cache)
    dout, dW1, db1, dgamma1, dbeta1 = conv_bn_relu_pool_backward(dout, conv1_cache)
    
    grads['W1'] = dW1 + self.reg*W1
    grads['b1'] = db1
    grads['gamma1'] = dgamma1
    grads['beta1'] = dbeta1
    
    grads['W2'] = dW2 + self.reg*W2
    grads['b2'] = db2
    grads['gamma2'] = dgamma2
    grads['beta2'] = dbeta2
    
    grads['W3'] = dW3 + self.reg*W3
    grads['b3'] = db3
    grads['gamma3'] = dgamma3
    grads['beta3'] = dbeta3
    
    grads['W4'] = dW4 + self.reg*W4
    grads['b4'] = db4
    grads['beta4'] = dbeta4
    grads['gamma4'] = dgamma4
    
    grads['W5'] = dW5 + self.reg*W5
    grads['b5'] = db5

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass
