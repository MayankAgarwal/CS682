def __affine_relu_dropout_forward(self, x, Wx, bx):
    x_affine, af_cache = affine_relu_forward(x, Wx, bx)
    x_dropout, dp_cache = dropout_forward(x_affine, self.dropout_param)
    
    cache = (af_cache, dp_cache)
    
    return x_dropout, cache


  def __affine_relu_dropout_backward(self, dout, cache):
    ad_cache, dp_cache = cache
    dout = dropout_backward(dout, dp_cache)
    dout, dW, db = affine_relu_backward(dout, af_cache)
    
    return dout, dW, db


  def __affine_bn_relu_forward(self, x_Wx, bx, gamma, beta, bn_params):
    
    x_affine, af_cache = affine_forward(x, Wx, bx)
    x_bn, bn_cache = batchnorm_forward(x_affine, gamma, beta, bn_params)
    x_relu, relu_cache = relu_forward(x_bn)
    
    cache = (af_cache, bn_cache, relu_cache)
    return x_relu, cache


  def __affine_batchnorm_relu_backward(self, dout, cache):
    af_cache, bn_cache, relu_cache = cache
    dout = relu_backward(dout, relu_cache)
    dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache)
    dout, dx, db = affine_backward(dout, af_cache)
    
    return dout, dx, db, dgamma, dbeta


  def __affine_bn_relu_dropout_forward(self, x, Wx, bx, gamma, beta, bn_params):
    x_affine, af_cache = affine_forward(x, Wx, bx)
    x_bn, bn_cache = batchnorm_forward(x_affine, gamma, beta, bn_params)
    x_relu, relu_cache = relu_forward(x_bn)
    x_dropout, dp_cache = dropout_forward(x_relu, self.dropout_param)
    
    cache = (af_cache, bn_cache, relu_cache, dp_cache)
    return x_dropout, cache


  def __affine_bn_relu_dropout_backward(self, dout, cache):
    
    af_cache, bn_cache, relu_cache, dp_cache = cache
    
    dout = dropout_backward(dout, dp_cache)
    dout = relu_backward(dout, relu_cache)
    dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache)
    dout, dx, db = affine_backward(dout, af_cache)
    
    return dout, dx, db, dgamma, dbeta
    
    