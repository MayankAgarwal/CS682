import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  ############################################################################# 
  
  N, D = X.shape
  C = W.shape[1]

  for i in xrange(N):
    
    total_prob = 0
    correct_prob = 0
    
    scores = np.dot(X[i,:], W)
    scores = np.exp(scores)
    total_prob = np.sum(scores)
    
    for j in xrange(C):
        
        dW_coeff = scores[j] / total_prob
        
        if j == y[i]:
            correct_prob = scores[j]
            dW_coeff -= 1
        
        dW[:, j] += dW_coeff * X[i, :].T
    
    loss += -1 * np.log(correct_prob/total_prob)

  loss /= N
  loss += 0.5 * reg * np.sum(W *W)

  dW /= N
  dW += (reg*W)

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  N, D = X.shape
    
  scores = X.dot(W)
  scores -= np.max(scores)  # for numerical stability
  scores = np.exp(scores)
  scores_sum = np.sum(scores, axis=1).reshape(N, 1)
  scores_softmax = np.divide(scores, scores_sum)
  
  loss = -1 * np.log(scores_softmax[range(N), y])
  loss = np.sum(loss) / N
  loss += 0.5 * reg * np.sum(W*W)

  dW_coeff = scores_softmax
  dW_coeff[range(N), y] -= 1
  dW = np.dot(X.T, dW_coeff)
    
  dW /= N
  dW += (reg * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

