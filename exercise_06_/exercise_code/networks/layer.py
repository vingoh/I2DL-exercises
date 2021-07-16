import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M)
    :param b: A numpy array of biases, of shape (M,)

    :return out: output, of shape (N, M)
    :return cache: (x, w, b)
    """
    out = None
    x_reshaped = np.reshape(x, (x.shape[0], -1))
    out = x_reshaped.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    :param dout: Upstream derivative, of shape (N, M)
    :param cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: A numpy array of biases, of shape (M,

    :return dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    :return dw: Gradient with respect to w, of shape (D, M)
    :return db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    dw = np.reshape(x, (x.shape[0], -1)).T.dot(dout)
    dw = np.reshape(dw, w.shape)

    db = np.sum(dout, axis=0, keepdims=False)

    dx = dout.dot(w.T)
    dx = np.reshape(dx, x.shape)
    return dx, dw, db


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        """
        Computes the forward pass for a sigmoid layer
        :param x: Inputs, of any shape

        :return out: Output, of the same shape as x
        :return cache: Cache, for backward computation, of the same shape as x
        """
        outputs = 1 / (1 + np.exp(-x))
        cache = outputs
        return outputs, cache

    def backward(self, dout, cache):
        """
        Computes the backward pass for a sigmoid layer.
        :param dout: Upstream derivative
        :param cache: Output of the forward pass

        :return: dx: the gradient w.r.t. input X
        """
        dx = None
        dx = dout * cache * (1 - cache)
        return dx


class Relu:
    def __init__(self):
        pass

    def forward(self, x):
        """
        Computes the forward pass for a ReLu layer.
        :param x: Inputs, of any shape

        :return out: Output, of the same shape as x
        :return cache: Cache, for backward computation, of the same shape as x
        """
        outputs = None
        cache = None

        outputs = np.maximum(x, 0)
        cache = x

        return outputs, cache

    def backward(self, dout, cache):
        """
        Computes backward pass for a ReLu layer.
        :param dout: Upstream derivative
        :param cache: Cache from forward() function, of the same
        shape than input to forward() function


        :return: dx: the gradient w.r.t. input X
        """
        dx = None

        x = cache
        dx = dout
        dx[x < 0] = 0

        return dx


class LeakyRelu:
    def __init__(self, slope=0.01):
        self.slope = slope

    def forward(self, x):
        """
        Computes forward pass for a LeakyReLu layer.
        :param x: Inputs, of any shape

        :return out: Output, of the same shape as x
        :return cache: Cache, for backward computation, of the same shape as x
        """
        outputs = np.zeros(x.shape)
        cache = np.zeros(x.shape)
        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass of LeakyRelu activation function          #
        ########################################################################

        pass
        cache = x
        outputs = x
        outputs[x < 0] = 0.01 * x[x < 0]

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return outputs, cache

    def backward(self, dout, cache):
        """
        Computes backward pass for a LeakyReLu layer.
        :param dout: Upstream derivative
        :param cache: Cache from forward() function, of the same
        shape than input to forward() function

        :return: dx: the gradient w.r.t. input X
        """
        dx = np.zeros((cache * dout).shape)
        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass of LeakyRelu activation function         #
        ########################################################################

        pass
        x = cache

        """
        dx = dout
        dx[x <= 0] = 0.01 * dout[x <= 0]
        """

        dx = np.ones_like(x)
        dx[x <= 0] = 0.01
        dx = dout * dx
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return dx


class Tanh:
    def __init__(self):
        pass

    def forward(self, x):
        """
        Computes the forward pass for a Tanh layer.
        :param x: Inputs, of any shape

        :return out: Output, of the same shape as x
        :return cache: Cache, for backward computation, of the same shape as x
        """
        outputs = np.ones(x.shape)
        cache = np.ones(x.shape)
        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass of Tanh activation function               #
        ########################################################################

        pass
        cache = x
        outputs = 2/(1 + np.exp(-2 * x)) - 1

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return outputs, cache

    def backward(self, dout, cache):
        """
        Computes the backward pass of a Tanh layer.
        :param dout: Upstream derivative
        :param cache: Output of the forward pass

        :return: dx: the gradient w.r.t. input X
        """
        dx = np.ones((cache * dout).shape)
        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass of Tanh activation function              #
        ########################################################################

        pass
        x = cache
        x = 2 * x
        sig_2x, _ = Sigmoid.forward(self, x)  #在Sigmoid类的forward前面加上@classmethod，这里就可以不用写self了
        dx = 4 * sig_2x * (1 - sig_2x) *dout

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return dx
