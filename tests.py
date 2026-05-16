import numpy as np

# ------------
# Test methods
# ------------

def test_rnncell_forward_1(RNNCell):
    layer = RNNCell(3, 4)
    layer.weight_ih = np.array([[-0.01, -0.02, -0.03],
                                [0.01, 0.02, 0.03],
                                [0.04, 0.05, 0.06],
                                [0.07, 0.08, 0.09]])
    layer.bias_ih = np.array([[-0.01, -0.02, 0.03, 0.04]])
    layer.weight_hh = np.array([[-0.08, -0.07, -0.06, -0.05],
                                [-0.04, -0.03, -0.02, -0.01],
                                [0., 0.01, 0.02, 0.03],
                                [0.04, 0.05, 0.06, 0.07]])
    layer.bias_hh = np.array([[0.01, 0.02, -0.03, -0.04]])
    x = np.array([[1., 2., 3.],
                  [4., 5., 6.]])
    h_prev = np.array([[1., 2., 3., 4.],
                    [3., 2., 1., 0.]])

    out = layer.forward(x, h_prev)
    return out

def test_rnncell_forward_2(RNNCell):
    layer = RNNCell(2, 5)
    layer.weight_ih = np.array([[-0.005, -5.,],
                                [0., 1.,],
                                [3., 4.,],
                                [6., 2.,],
                                [0., 0.]])
    layer.bias_ih = np.array([[-0.0002, -2., 1., 1., 0.]])
    layer.weight_hh = np.array([[-5., -4., -3., -2., -1.],
                                [-4., -3., -2., -1., 0.],
                                [0., 1., 2., 3., 0.],
                                [4., 5., 6., 7., 6.],
                                [7., 8., 9., -0.25, 0.25]])
    layer.bias_hh = np.array([[0.001, 2., -3., -4., 0.]])
    x = np.array([[0.0005, 2.],
                  [-0.152, -0.025],
                  [0., -1.]])
    h_prev = np.array([[0.02, 0.0005, 0.15, 0.45, 0.25],
                       [-0.001, 2., 1., 0., -0.25],
                       [1., -2., -5., -10., 0.]])

    out = layer.forward(x, h_prev)
    return out

def test_rnncell_forward_3(RNNCell):
    layer = RNNCell(5, 2)
    layer.weight_ih = np.array([[-0.005, -0.01, 0.05, 0.15, 0.222],
                                [0.151, 1.10, 0.015, -0.002, 0.051]])
    layer.bias_ih = np.array([[0.11, -0.15]])
    layer.weight_hh = np.array([[-0.152, 0.112],
                                [-0.002, -1.01]])
    layer.bias_hh = np.array([[0.001, 2.]])
    x = np.array([[-0.001, -2.015, -0.125, -0.001, 1.025],
                  [0.122, -0.128, -0.0002, -0.01, 0.15],
                  [-0.01, -0.002, 0.0152, 0.123, 0.231],
                  [-0.821, 0.999, 0.251, 0.331, 0.025]])
    h_prev = np.array([[0.02, 0.0005],
                       [-0.001, 2.],
                       [1., -2.],
                       [0.025, 0.152]])

    out = layer.forward(x, h_prev)
    return out

def test_rnncell_backward_1(RNNCell):
    layer = RNNCell(3, 4)
    layer.weight_ih = np.array([[-0.01, -0.02, -0.03],
                                [0.01, 0.02, 0.03],
                                [0.04, 0.05, 0.06],
                                [0.07, 0.08, 0.09]])
    layer.bias_ih = np.array([[-0.01, -0.02, 0.03, 0.04]])
    layer.weight_hh = np.array([[-0.08, -0.07, -0.06, -0.05],
                                [-0.04, -0.03, -0.02, -0.01],
                                [0., 0.01, 0.02, 0.03],
                                [0.04, 0.05, 0.06, 0.07]])
    layer.bias_hh = np.array([[0.01, 0.02, -0.03, -0.04]])
    h_prev = np.array([[1., 2., 3., 4.],
                    [3., 2., 1., 0.]])
    
    grad = np.array([[ 0.402888  ,  0.01693988,  0.04669028,  0.54219921],
                     [-0.88253697,  0.11543817, -0.70723666, -0.69393529]])
    h_prev_l = np.array([[-0.62162275, -0.96164911,  0.21242244],
                         [ 0.27069328, -0.83331687, -0.6866582 ]])
    h_prev_t = np.array([[ 0.33983293,  0.93178091,  0.69990522, -0.15981829],
                         [ 0.12239934, -0.56387259, -0.9556718 , -0.31575391]])

    dx, dh = layer.backward(grad, h_prev, h_prev_l, h_prev_t)
    return dx, dh, layer.grad_weight_ih, layer.grad_weight_hh, layer.grad_bias_ih, layer.grad_bias_hh


def test_rnncell_backward_2(RNNCell):
    layer = RNNCell(2, 5)
    layer.weight_ih = np.array([[-0.005, -5.,],
                                [0., 1.,],
                                [3., 4.,],
                                [6., 2.,],
                                [0., 0.]])
    layer.bias_ih = np.array([[-0.0002, -2., 1., 1., 0.]])
    layer.weight_hh = np.array([[-5., -4., -3., -2., -1.],
                                [-4., -3., -2., -1., 0.],
                                [0., 1., 2., 3., 0.],
                                [4., 5., 6., 7., 6.],
                                [7., 8., 9., -0.25, 0.25]])
    layer.bias_hh = np.array([[0.001, 2., -3., -4., 0.]])

    h_prev = np.array([[0.02, 0.0005, 0.15, 0.45, 0.25],
                       [-0.001, 2., 1., 0., -0.25],
                       [1., -2., -5., -10., 0.]])
    
    grad = np.array([[-0.9207447 , -0.0299716 ,  0.93634652, -0.75164411, -0.73439064],
       [-0.09640495, -0.87471139,  0.72295136,  0.8407652 ,  0.94054401],
       [ 0.08871341, -0.30203741,  0.57196294, -0.05680726, -0.59568463]])
    h_prev_l = np.array([[-0.66812673,  0.54131346],
       [-0.09802792,  0.56934838],
       [-0.0081655 ,  0.59356404]])
    h_prev_t = np.array([[ 0.72465154,  0.04605758,  0.44489962, -0.81581832, -0.46412398],
       [-0.11491536, -0.90658557, -0.06506369,  0.93885842, -0.88547036],
       [-0.86381171, -0.59843711, -0.63817132, -0.06270253,  0.22630636]])

    dx, dh = layer.backward(grad, h_prev, h_prev_l, h_prev_t)
    return dx, dh, layer.grad_weight_ih, layer.grad_weight_hh, layer.grad_bias_ih, layer.grad_bias_hh

def test_rnncell_backward_3(RNNCell):
    layer = RNNCell(5, 2)
    layer.weight_ih = np.array([[-0.005, -0.01, 0.05, 0.15, 0.222],
                                [0.151, 1.10, 0.015, -0.002, 0.051]])
    layer.bias_ih = np.array([[0.11, -0.15]])
    layer.weight_hh = np.array([[-0.152, 0.112],
                                [-0.002, -1.01]])
    layer.bias_hh = np.array([[0.001, 2.]])
    h_prev = np.array([[0.02, 0.0005],
                       [-0.001, 2.],
                       [1., -2.],
                       [0.025, 0.152]])
    
    grad = np.array([[-0.19521881, -0.82817359],
       [-0.52215925, -0.16664658],
       [ 0.82267516, -0.52842318],
       [-0.78720939, -0.11186485]])
    h_prev_l = np.array([[-0.10070971, -0.14045025, -0.31836873,  0.22785118, -0.40093625],
       [ 0.01313353, -0.05711794,  0.37709152, -0.80443835,  0.53103029],
       [ 0.84935059,  0.00150989, -0.76366634,  0.10226739, -0.83378106],
       [-0.71682395,  0.10407237,  0.84127296,  0.25157057, -0.28898927]])
    h_prev_t = np.array([[-0.28169733,  0.05609386],
       [ 0.66147383, -0.41036188],
       [ 0.51908581, -0.31614092],
       [-0.337796  , -0.46155793]])

    dx, dh = layer.backward(grad, h_prev, h_prev_l, h_prev_t)
    return dx, dh, layer.grad_weight_ih, layer.grad_weight_hh, layer.grad_bias_ih, layer.grad_bias_hh

def test_rnn_forward_1(RNN):
    rnn = RNN(3, 4, num_layers=2)
    
    rnn.layers[0].weight_ih = np.array([[-0.01, -0.02, -0.03],
                                [0.01, 0.02, 0.03],
                                [0.04, 0.05, 0.06],
                                [0.07, 0.08, 0.09]])
    rnn.layers[0].bias_ih = np.array([[-0.01, -0.02, 0.03, 0.04]])
    
    rnn.layers[0].weight_hh = np.array([[-0.08, -0.07, -0.06, -0.05],
                                [-0.04, -0.03, -0.02, -0.01],
                                [0., 0.01, 0.02, 0.03],
                                [0.04, 0.05, 0.06, 0.07]])
    rnn.layers[0].bias_hh = np.array([[0.01, 0.02, -0.03, -0.04]])
    
    rnn.layers[1].weight_ih = np.array([[-0.48600084,  0.56108125,  0.60350991, -0.12398511],
       [-0.61426125,  0.74533839,  0.28829775,  0.39298129],
       [-0.23660597,  0.60180463, -0.17925026, -0.91010067],
       [ 0.32842213,  0.88810569,  0.47370219, -0.02676785]])
    rnn.layers[1].bias_ih = np.array([[ 0.80336632,  0.5080941 , -0.02041526, -0.23457551]])
    rnn.layers[1].weight_hh = np.array([[-0.66123341,  0.89060331,  0.52273452,  0.51176128],
       [ 0.56754451, -0.00703578,  0.7770004 ,  0.27559471],
       [ 0.55518779, -0.73064323,  0.9099351 ,  0.57656225],
       [ 0.36056783, -0.05294811,  0.47201524, -0.4093564 ]])
    rnn.layers[1].bias_hh = np.array([[-0.03656752, -0.7702112 ,  0.43095679,  0.89191036]])
    
    x = np.array([[[ 0.54963061, -0.05740221, -0.18798255],
        [-0.05139327,  0.31849613, -0.71975814],
        [ 0.25741578,  0.5026123 ,  0.64851701],
        [ 0.15833562,  0.45382923,  0.17070994],
        [-0.02513259,  0.7292614 ,  0.84537154]],

       [[ 0.96812713, -0.77282445, -0.95171846],
        [ 0.38731339, -0.83533687,  0.99131245],
        [ 0.04599233, -0.95944108,  0.23168402],
        [ 0.16541892,  0.13714912, -0.95989072],
        [ 0.51472085,  0.37945332,  0.4281448 ]]])
    h_0 = np.array([[[-0.71008325, -0.07277072,  0.78528585, -0.35171659],
        [ 0.21250086,  0.90839637, -0.7255993 ,  0.53047081]],

       [[-0.86511414, -0.94116341, -0.36088932,  0.69915346],
        [-0.53193134,  0.29990252, -0.27214273, -0.72523786]]])
    
    out, hiddens = rnn.forward(x, h_0)
    return out, hiddens

def test_rnn_forward_2(RNN):
    rnn = RNN(2, 5, num_layers=1)
    
    rnn.layers[0].weight_ih = np.array([[-0.005, -5.,],
                                [0., 1.,],
                                [3., 4.,],
                                [6., 2.,],
                                [0., 0.]])
    rnn.layers[0].bias_ih = np.array([[-0.0002, -2., 1., 1., 0.]])
    rnn.layers[0].weight_hh = np.array([[-5., -4., -3., -2., -1.],
                                [-4., -3., -2., -1., 0.],
                                [0., 1., 2., 3., 0.],
                                [4., 5., 6., 7., 6.],
                                [7., 8., 9., -0.25, 0.25]])
    rnn.layers[0].bias_hh = np.array([[0.001, 2., -3., -4., 0.]])
    x = np.array([[[-0.37172187,  0.75289209],
            [-0.74297175, -0.02525347],
            [ 0.53391313, -0.67597502],
            [-0.7237738 , -0.26039564],
            [-0.80266149,  0.16000827]],

        [[ 0.21733297,  0.63915498],
            [-0.29721541, -0.99389827],
            [-0.41202637, -0.26365921],
            [-0.12466216, -0.60468756],
            [ 0.93159041, -0.78198452]],

        [[-0.87435205, -0.84892116],
            [ 0.46827638,  0.34440284],
            [ 0.88852153,  0.09396869],
            [-0.9570812 , -0.00129956],
            [ 0.24959137, -0.85093966]]])
    h_0 = np.array([[0.02, 0.0005, 0.15, 0.45, 0.25],
                       [-0.001, 2., 1., 0., -0.25],
                       [1., -2., -5., -10., 0.]])
    
    out, hiddens = rnn.forward(x, h_0)
    return out, hiddens

def test_rnn_forward_3(RNN):
    rnn = RNN(5, 2, num_layers=3)    
    rnn.layers[0].weight_ih = np.array([[-0.005, -0.01, 0.05, 0.15, 0.222],
                                        [0.151, 1.10, 0.015, -0.002, 0.051]])
    rnn.layers[0].bias_ih = np.array([[0.11, -0.15]])
    rnn.layers[0].weight_hh = np.array([[-0.152, 0.112],
                                        [-0.002, -1.01]])
    rnn.layers[0].bias_hh = np.array([[0.001, 2.]])

    
    rnn.layers[1].weight_ih = np.array([[ 0.00372146, -0.82457555],
       [ 0.87680038, -0.46820705]])
    rnn.layers[1].bias_ih = np.array([ 0.6543994 , -0.49759433])
    rnn.layers[1].weight_hh = np.array([[-0.49185334, -0.22758022],
       [ 0.14179349,  0.74545256]])
    rnn.layers[1].bias_hh = np.array([ 0.10898453, -0.62921271])
    
    rnn.layers[2].weight_ih = np.array([[ 0.43795292, -0.39165178],
       [-0.43341205,  0.97362009]])
    rnn.layers[2].bias_ih = np.array([-0.30573836,  0.65735053])
    rnn.layers[2].weight_hh = np.array([[-0.03211809,  0.92450192],
       [-0.8617065 ,  0.03840494]])
    rnn.layers[2].bias_hh = np.array([-0.77441866,  0.05598034])
    
    
    x = np.array([[[-0.17857984, -0.93947819, -0.67213643, -0.69538802,
          0.20826295],
        [ 0.70933672,  0.55675034, -0.31722017,  0.96747612,
          0.69028081],
        [ 0.92119125, -0.13467066, -0.35187102, -0.20076277,
         -0.72281168]],

       [[ 0.36549078, -0.40412135, -0.50507736, -0.96217543,
         -0.68987774],
        [ 0.47556309, -0.21759229,  0.88103497, -0.77756477,
         -0.47552185],
        [-0.21578696,  0.55293293,  0.68532053,  0.78959926,
         -0.56555195]]])
    h_0 = np.array([[[-0.81155394, -0.67203904],
        [ 0.21009179, -0.28347711]],

       [[ 0.43134709,  0.19829147],
        [-0.20346234,  0.59861199]],

       [[-0.73887404, -0.843644  ],
        [-0.36426267,  0.3433386 ]]])
    
    out, hiddens = rnn.forward(x, h_0)
    return out, hiddens

def test_rnn_backward_1(RNN):
    rnn = RNN(3, 4, num_layers=2)
    
    rnn.layers[0].weight_ih = np.array([[-0.01, -0.02, -0.03],
                                [0.01, 0.02, 0.03],
                                [0.04, 0.05, 0.06],
                                [0.07, 0.08, 0.09]])
    rnn.layers[0].bias_ih = np.array([[-0.01, -0.02, 0.03, 0.04]])
    
    rnn.layers[0].weight_hh = np.array([[-0.08, -0.07, -0.06, -0.05],
                                [-0.04, -0.03, -0.02, -0.01],
                                [0., 0.01, 0.02, 0.03],
                                [0.04, 0.05, 0.06, 0.07]])
    rnn.layers[0].bias_hh = np.array([[0.01, 0.02, -0.03, -0.04]])
    
    rnn.layers[1].weight_ih = np.array([[-0.48600084,  0.56108125,  0.60350991, -0.12398511],
       [-0.61426125,  0.74533839,  0.28829775,  0.39298129],
       [-0.23660597,  0.60180463, -0.17925026, -0.91010067],
       [ 0.32842213,  0.88810569,  0.47370219, -0.02676785]])
    rnn.layers[1].bias_ih = np.array([[ 0.80336632,  0.5080941 , -0.02041526, -0.23457551]])
    rnn.layers[1].weight_hh = np.array([[-0.66123341,  0.89060331,  0.52273452,  0.51176128],
       [ 0.56754451, -0.00703578,  0.7770004 ,  0.27559471],
       [ 0.55518779, -0.73064323,  0.9099351 ,  0.57656225],
       [ 0.36056783, -0.05294811,  0.47201524, -0.4093564 ]])
    rnn.layers[1].bias_hh = np.array([[-0.03656752, -0.7702112 ,  0.43095679,  0.89191036]])
    
    x = np.array([[[ 0.54963061, -0.05740221, -0.18798255],
        [-0.05139327,  0.31849613, -0.71975814],
        [ 0.25741578,  0.5026123 ,  0.64851701],
        [ 0.15833562,  0.45382923,  0.17070994],
        [-0.02513259,  0.7292614 ,  0.84537154]],

       [[ 0.96812713, -0.77282445, -0.95171846],
        [ 0.38731339, -0.83533687,  0.99131245],
        [ 0.04599233, -0.95944108,  0.23168402],
        [ 0.16541892,  0.13714912, -0.95989072],
        [ 0.51472085,  0.37945332,  0.4281448 ]]])
    h_0 = np.array([[[-0.71008325, -0.07277072,  0.78528585, -0.35171659],
        [ 0.21250086,  0.90839637, -0.7255993 ,  0.53047081]],

       [[-0.86511414, -0.94116341, -0.36088932,  0.69915346],
        [-0.53193134,  0.29990252, -0.27214273, -0.72523786]]])
    
    rnn.forward(x, h_0)
    
    grad = np.array([[-0.1, 0.2, -0.3, 0.4],
                     [-0.4, 0.5, -0.6, 0.7]])
    
    dx, dh_0 = rnn.backward(grad)
    
    # Gather all the gradients in a single list
    layer_gradients = []
    for n in range(rnn.num_layers):
        layer_gradients.append([rnn.layers[n].grad_weight_ih, rnn.layers[n].grad_weight_hh, rnn.layers[n].grad_bias_ih, rnn.layers[n].grad_bias_hh])
    
    return dx, dh_0, layer_gradients

def test_rnn_backward_2(RNN):
    rnn = RNN(2, 5, num_layers=1)
    
    rnn.layers[0].weight_ih = np.array([[-0.005, -5.,],
                                [0., 1.,],
                                [3., 4.,],
                                [6., 2.,],
                                [0., 0.]])
    rnn.layers[0].bias_ih = np.array([[-0.0002, -2., 1., 1., 0.]])
    rnn.layers[0].weight_hh = np.array([[-5., -4., -3., -2., -1.],
                                [-4., -3., -2., -1., 0.],
                                [0., 1., 2., 3., 0.],
                                [4., 5., 6., 7., 6.],
                                [7., 8., 9., -0.25, 0.25]])
    rnn.layers[0].bias_hh = np.array([[0.001, 2., -3., -4., 0.]])
    x = np.array([[[-0.37172187,  0.75289209],
            [-0.74297175, -0.02525347],
            [ 0.53391313, -0.67597502],
            [-0.7237738 , -0.26039564],
            [-0.80266149,  0.16000827]],

        [[ 0.21733297,  0.63915498],
            [-0.29721541, -0.99389827],
            [-0.41202637, -0.26365921],
            [-0.12466216, -0.60468756],
            [ 0.93159041, -0.78198452]],

        [[-0.87435205, -0.84892116],
            [ 0.46827638,  0.34440284],
            [ 0.88852153,  0.09396869],
            [-0.9570812 , -0.00129956],
            [ 0.24959137, -0.85093966]]])
    h_0 = np.array([[0.02, 0.0005, 0.15, 0.45, 0.25],
                       [-0.001, 2., 1., 0., -0.25],
                       [1., -2., -5., -10., 0.]])
    
    rnn.forward(x, h_0)
    
    grad = np.array([[ 0.93211174, -0.38224902, -0.48032845,  0.74022526, -0.24474702],
       [ 0.0575357 ,  0.86414712,  0.30018032,  0.95602853,  0.09027887],
       [-0.62221448,  0.56082445, -0.75347751,  0.4888543 ,  0.80749852]])
    
    dx, dh_0 = rnn.backward(grad)
    
    # Gather all the gradients in a single list
    layer_gradients = []
    for n in range(rnn.num_layers):
        layer_gradients.append([rnn.layers[n].grad_weight_ih, rnn.layers[n].grad_weight_hh, rnn.layers[n].grad_bias_ih, rnn.layers[n].grad_bias_hh])
    
    return dx, dh_0, layer_gradients

def test_rnn_backward_3(RNN):
    rnn = RNN(5, 2, num_layers=3)    
    rnn.layers[0].weight_ih = np.array([[-0.005, -0.01, 0.05, 0.15, 0.222],
                                        [0.151, 1.10, 0.015, -0.002, 0.051]])
    rnn.layers[0].bias_ih = np.array([[0.11, -0.15]])
    rnn.layers[0].weight_hh = np.array([[-0.152, 0.112],
                                        [-0.002, -1.01]])
    rnn.layers[0].bias_hh = np.array([[0.001, 2.]])

    
    rnn.layers[1].weight_ih = np.array([[ 0.00372146, -0.82457555],
       [ 0.87680038, -0.46820705]])
    rnn.layers[1].bias_ih = np.array([ 0.6543994 , -0.49759433])
    rnn.layers[1].weight_hh = np.array([[-0.49185334, -0.22758022],
       [ 0.14179349,  0.74545256]])
    rnn.layers[1].bias_hh = np.array([ 0.10898453, -0.62921271])
    
    rnn.layers[2].weight_ih = np.array([[ 0.43795292, -0.39165178],
       [-0.43341205,  0.97362009]])
    rnn.layers[2].bias_ih = np.array([-0.30573836,  0.65735053])
    rnn.layers[2].weight_hh = np.array([[-0.03211809,  0.92450192],
       [-0.8617065 ,  0.03840494]])
    rnn.layers[2].bias_hh = np.array([-0.77441866,  0.05598034])
    
    
    x = np.array([[[-0.17857984, -0.93947819, -0.67213643, -0.69538802,
          0.20826295],
        [ 0.70933672,  0.55675034, -0.31722017,  0.96747612,
          0.69028081],
        [ 0.92119125, -0.13467066, -0.35187102, -0.20076277,
         -0.72281168]],

       [[ 0.36549078, -0.40412135, -0.50507736, -0.96217543,
         -0.68987774],
        [ 0.47556309, -0.21759229,  0.88103497, -0.77756477,
         -0.47552185],
        [-0.21578696,  0.55293293,  0.68532053,  0.78959926,
         -0.56555195]]])
    h_0 = np.array([[[-0.81155394, -0.67203904],
        [ 0.21009179, -0.28347711]],

       [[ 0.43134709,  0.19829147],
        [-0.20346234,  0.59861199]],

       [[-0.73887404, -0.843644  ],
        [-0.36426267,  0.3433386 ]]])
    
    rnn.forward(x, h_0)
    
    grad = np.array([[ 0.65155603, -0.78930043],
       [ 0.11431596,  0.36581844]])
    
    dx, dh_0 = rnn.backward(grad)
    
    # Gather all the gradients in a single list
    layer_gradients = []
    for n in range(rnn.num_layers):
        layer_gradients.append([rnn.layers[n].grad_weight_ih, rnn.layers[n].grad_weight_hh, rnn.layers[n].grad_bias_ih, rnn.layers[n].grad_bias_hh])
    
    return dx, dh_0, layer_gradients

# ---------------------
# General methods below
# ---------------------

def compare_to_answer(user_output, answer, test_name=None):
    # Check that the object type of user's answer is correct
    if not check_types_same(user_output, answer, test_name):
        return False
    # Check that the shape of the user's answer matches the expected shape
    if not check_shapes_same(user_output, answer, test_name):
        return False
    # Check that the values of the user's answer matches the expected values
    if not check_values_same(user_output, answer, test_name):
        return False
    # If passed all the above tests, return True
    return True

def check_types_same(user_output, answer, test_name=None):
    try:
        assert isinstance(user_output, type(answer))
    except Exception as e:
        if test_name:
            print(f'Incorrect object type for {test_name}.')
        print("Your output's type:", type(user_output))
        print("Expected type:", type(answer))
        return False
    return True

def check_shapes_same(user_output, answer, test_name=None):
    try:
        assert user_output.shape == answer.shape
    except Exception as e:
        if test_name:
            print(f'Incorrect shape for {test_name}.')
        print('Your shape:', user_output.shape)
        print('Your values:\n', user_output)
        print('Expected shape:', answer.shape)
        return False
    return True

def check_values_same(user_output, answer, test_name=None):
    try:
        assert np.allclose(user_output, answer)
    except Exception as e:
        if test_name:
            print(f'Incorrect values for {test_name}.')
        print('Your values:\n', user_output)
        return False
    return True
