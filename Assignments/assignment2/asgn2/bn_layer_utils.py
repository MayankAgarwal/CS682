from asgn2.layers import *
from asgn2.layer_utils import *

def affine_relu_dropout_forward(x, Wx, bx, dropout_param):
    x_affine, af_cache = affine_relu_forward(x, Wx, bx)
    x_dropout, dp_cache = dropout_forward(x_affine, dropout_param)

    cache = (af_cache, dp_cache)

    return x_dropout, cache


def affine_relu_dropout_backward(dout, cache):
    af_cache, dp_cache = cache
    dout = dropout_backward(dout, dp_cache)
    dout, dW, db = affine_relu_backward(dout, af_cache)

    return dout, dW, db


def affine_bn_relu_forward(x, Wx, bx, gamma, beta, bn_params):

    x_affine, af_cache = affine_forward(x, Wx, bx)
    x_bn, bn_cache = batchnorm_forward(x_affine, gamma, beta, bn_params)
    x_relu, relu_cache = relu_forward(x_bn)

    cache = (af_cache, bn_cache, relu_cache)
    return x_relu, cache


def affine_batchnorm_relu_backward(dout, cache):
    af_cache, bn_cache, relu_cache = cache
    dout = relu_backward(dout, relu_cache)
    dout, dgamma, dbeta = batchnorm_backward_alt(dout, bn_cache)
    dout, dx, db = affine_backward(dout, af_cache)

    return dout, dx, db, dgamma, dbeta


def affine_bn_relu_dropout_forward(x, Wx, bx, gamma, beta, bn_params, dropout_param):
    x_affine, af_cache = affine_forward(x, Wx, bx)
    x_bn, bn_cache = batchnorm_forward(x_affine, gamma, beta, bn_params)
    x_relu, relu_cache = relu_forward(x_bn)
    x_dropout, dp_cache = dropout_forward(x_relu, dropout_param)

    cache = (af_cache, bn_cache, relu_cache, dp_cache)
    return x_dropout, cache


def affine_bn_relu_dropout_backward(dout, cache):

    af_cache, bn_cache, relu_cache, dp_cache = cache

    dout = dropout_backward(dout, dp_cache)
    dout = relu_backward(dout, relu_cache)
    dout, dgamma, dbeta = batchnorm_backward_alt(dout, bn_cache)
    dout, dx, db = affine_backward(dout, af_cache)

    return dout, dx, db, dgamma, dbeta

def conv_bn_relu_pool_forward(x, w, b, gamma, beta, conv_param, pool_param, bn_param):
  """
  Convenience layer that performs a convolution, a spatial BN,  a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  b, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  s, relu_cache = relu_forward(b)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache, bn_cache)
  return out, cache


def conv_bn_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache, bn_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  db, dgamma, dbeta = spatial_batchnorm_backward(da, bn_cache)
  dx, dw, db = conv_backward_fast(db, conv_cache)
  return dx, dw, db, dgamma, dbeta
