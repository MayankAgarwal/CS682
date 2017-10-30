from asgn2.layers import *
from asgn2.layer_utils import *

def affine_relu_dropout_forward(x, Wx, bx, dropout_param):
    x_affine, af_cache = affine_relu_forward(x, Wx, bx)
    x_dropout, dp_cache = dropout_forward(x_affine, dropout_param)

    cache = (af_cache, dp_cache)

    return x_dropout, cache


def affine_relu_dropout_backward(dout, cache):
    ad_cache, dp_cache = cache
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
    dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache)
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
    dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache)
    dout, dx, db = affine_backward(dout, af_cache)

    return dout, dx, db, dgamma, dbeta

