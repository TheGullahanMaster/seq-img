# seq-img
A script created for sequential generation of images

Simply run train.py, and follow instructions

Experiments show that columnwise learning is the fastest/best, tied with rowwise. Classic Patch mode learns global features faster, but inner content takes far longer to converge (slightly faster if LSTM is used, but those have their own issues)

UPDATE:
Applying grad norm clipping has largely equalized all three patch modes into viability.

# Model types:
1) LSTM - An nn.LSTM setup with patching in mind. It performs reasonably well, but is slow to converge fully
2) GRU - WIP, absolutely terrible performance, learning rate most likely requires a tune-up
3) MinGRU (implementation by Lucidrains, modified for patched inputs) - Works decently well, but is outcompeted by better options. Still figuring out the sampling strategy
4) MLP - A highly flexible MLP with selectable positional embedding types, activation functions, normalization type and residual type. Can perform very well on simpler datasets, but struggles on more complex datasets, as expected. Sine, Growing Sine and growing Cosine, SELU, Sigmoid and Tanh have their appropriate initialization, the rest is initialized with Xavier, ReLU gain (as the other functions are mostly ReLU variants outside of some exceptions).
5) Modified GPT-2 style transformer. Selectable positional embedding types. Very good performance even on more complex datasets. Loves to overfit if dataset is undersized.
6) ALiBi Transformer. Just like GPT-2, but with no pos embeds and the ALiBi attention allowing for attention to learn positional context. Very good performance, a little below GPT-2 but not by much. Extrapolable qualities of ALiBi not very pronounced as images have the same size every time.
7) "Modernized" GPT-2 transformer, featuring Dropout and DropPath, Rotary Pos embeddings and Trainable layer scaling. Very good performance, sometimes outperforms GPT-2, tends to produce noisier outputs. Very fast initial convergence
8) Graph Neural Network - Very WIP, nan loss ocurs instantly upon starting training. Will fix eventually.
9) Mamba - WIP, exremely large intial losses. It does not perform well, but the issue is most likely in the implementation, not the architecture itself.
10) Causal Convnet - A very basic causal convnet. Poor to decent performance, will need to tweak archiutecture.



# Pos embeds types
(0) NoPos - No positional information. MLP and GPT-2 work without these. MLP is quite a bit worse without them, GPT-2 works fine, if a little worse than with pos embeds. The research into nopos transformers is where this option got it's name from
(1) Sinusoidal Positional Embeddings - Original transformer's sinusoidal pos embeds. Don't generalize well outside of seq_len, which is not really an issue here.
(2) Trainable positional embeddings - A trainable positional embedding. Each patch gets "hidden_dim" positional information. The values are tweaked via backpropagation
# Activation functions - This and the ones further down are only for the MLP so far. I do eventually want to include these in the transformers and MinGRU
(0) Sigmoid - Basic sigmoid. Performs badly
(1) Hyperbolic Tangent - Tanh, performs better than Sigmoid
(2) Sine - Has it's own SIREN-like intialization of layers. Not much testing done yet, but presumably preferable in periodic image generation
(3) Rectified Linear Unit - ReLU. Extremely fast, can suffer from dying ReLU problem. Performs pretty well
(4) Scaled Exponentional Linear Unit - SELU, has it's own init (LeCun), performs best WITHOUT residuals. Performs well, can train fairly deep MLPs with this one
(5) Cone Activation Function - (1 - |x - 1|), Cone shaped activation function. Supposedly it helps NNs to split datapoints into hyperstrips. Can work well, but can also be finicky. More testing needed.
(6) Gombertz Linear Unit - GoLU (x * exp(-exp(-x))). It has clamping on values below -5 due to numerical issues caused by the negative exp causing autograd to throw out NaNs. Works pretty well.
(7) Parametric ReLU - Leaky ReLU with a trainable slope. Performs very well.
(8) LeakyRELU 0.01 - Original LeakyReLU. Comparable to ReLU, but without the dying neurons
(9) LeakyReLU 0.2 - Leaky ReLU with a slope of 0.2. Faster training but potentionally weaker performance. It depends on the dataset being trained really.
(10) LeakyReLU -1 - Leaky ReLU with a slope of -1, essentially turning it into the absolute value function. Can be unstable, but could also work as a cone shaped activation function
(11) Sigmoid Linear Unit - SiLU. A proven ReLU alternative. Performs very well in here as well
(12) Gaussian Error Linear Unit - GELU, fantastic performance.
(13) TeLU - Here defined as x * torch.tanh(torch.exp(torch.clamp(x, max=5))). Performs worse than GoLU and GELU, but still performs well.
(14) The Square - Imagine a square, where each corner corresponds to the activation fucntions Tanh, TeLU, Sine and Cone. The activation has a trainable cooridnate system, with each activation being assigned a corner. Moving the x,y coordinates (done via backprop) essentially interpolates between the four. Slightly higher computational overhead (it does in fact run all four everytime to interpolate), performs decently
(15) Growing sine - x * sin(x). Seems to handle deeper nonresidual networks better than vanilla sine
(16) Growing cosine - x * cos(x). Similar to growing sine
(17) HeLU - ReLU with custom backward pass. In this, the forward function is the ReLU, the backward pass however shifts the threshold slightly to the left (set by an alpha parameter), minimizing dying ReLU problem. Currently has an alpha of 0.05. Very solid activation function, can handle surprisingly deep nonresidual networks.
(18) Sign activation - Just the sign activation. The pytorch's autograd is doing some tricks to get training underway, but it still underperforms a lot
(19) Adaptive Gated Linear Unit (AGLU) - Supposed to adapt the self-gating mechanism, interpolationg between sigmoid (as in SiLU) and Gumbel function. Fairly comparable to SiLU
(20) Trainable piecewise linear unit - 32 breakpoints. It has traianble breakpoints with trainable slopes. Initialized as a linear function. Can perform really well theoretically, but takes quite a bit to arrive there. Different init's could lead to better results. Can be computationally heavy
(21) Parabolic Cone - (x * (2 - x)). Like Cone, but smooth. Initially unstable, as the values outside the bump grow exponentionally, but eventually straightens itself out. Performs slightly better than cone, fully differentiable 
(22) No activation (Linear). Simply an Identity function. It's just here to compare linear networks with nonlinear ones.
# Type of activation
(0) Normal - Basic activation function
(1) GLU Variant - This one will take the previously chosen activation and use it as the gating mechanism in the GLU
# Residual Type
(0) No residuals - Simple feedforward system. Troublesome at deeper networks, but less "artifacty" than residuals.
(1) Highway connections - Not tested much yet. Predecessors to Residuals. here to compare with the current Residual systems
(2) Standard residual - x + layer(x). Standard issue residual connection. Improves gradient flow
(3) ReZero Resdiual - x + layer(x) * gamma. Gamma is initialized to 0.0 per layer, making the network start as an identity function. Improves even further beyond residual, at it's very worst it still functions like a trainable layer sclaing system/normalization
# Normalization technique
(0) No norm - No normalization applied. Preferable option for SELU.
(1) Instance Normalization - Has better handling of small batch sizes than batch norm, but is currently only used in AdaIN-like systems (as far as i can tell). Still usually better than no norm.
(2) Batch Normalization - Very good if your batch size can get above ~8-12 examples per batch (strong hardware or small images), otherwise will falter.
(3) RMSNorm - used in certain arhitectures where subtracting the mean is not preferable. Basically a simpler layernorm. Excels in some tasks
(4) Layernorm - The most widely used normalization, especially in NLP with transformers and RNNs. Performs well, but certain activations can underperform (most notably the Cone) with this applied, as it warps the AF curve. Linear networks with Layernorm can actually learn simple nonlinear problems due to it's affine transforms introducing a slight nonlinearity
