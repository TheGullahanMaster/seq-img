import os
import glob
import json
import random
import math
from PIL import Image, ImageFilter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from lamb import *  # Provides TeLU and RAdamScheduleFree – ensure lamb.py is available
from torch.nn import Linear, Module, Identity
from torch.jit import script
from transformers import GPT2LMHeadModel, GPT2Config
import os
import glob
import json
import random
import math
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from lamb import *  # Provides TeLU, GoLU and RAdamScheduleFree – ensure lamb.py is available
from torch.nn import Linear, Module, Identity
from torch.jit import script
from transformers import GPT2LMHeadModel, GPT2Config
#############################################
# Custom Dataset Class with Patch Mode
#############################################
class PatchImageDataset(Dataset):
    def __init__(self, image_dir, image_size, patch_mode, patch_size=None, patch_height=None, patch_width=None):
        """
        Args:
            image_dir (str): Directory containing images.
            image_size (int): Size to which images are resized (assumed square).
            patch_mode (int): 
                0 for classic patches (grid), 
                1 for rowwise patches (each patch covers a horizontal strip), 
                2 for columnwise patches (each patch covers a vertical strip).
            patch_size (int): Fallback if individual dimensions are not provided.
            patch_height (int): For mode 0 and 1: the patch height.
            patch_width (int): For mode 0 and 2: the patch width.
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.patch_mode = patch_mode  # 0, 1, or 2

        if self.patch_mode == 0:
            # For grid patches, use patch_height and patch_width; fall back to patch_size if needed.
            self.patch_height = patch_height if patch_height is not None else patch_size
            self.patch_width = patch_width if patch_width is not None else patch_size
            self.patch_vector_dim = self.patch_height * self.patch_width * 3
        elif self.patch_mode == 1:  # rowwise: patch covers full width; height is given
            self.patch_height = patch_height if patch_height is not None else patch_size
            self.patch_vector_dim = self.image_size * self.patch_height * 3
        elif self.patch_mode == 2:  # columnwise: patch covers full height; width is given
            self.patch_width = patch_width if patch_width is not None else patch_size
            self.patch_vector_dim = self.image_size * self.patch_width * 3

        # Recursively collect image paths.
        self.image_paths = []
        for ext in ['png', 'jpg', 'jpeg', 'bmp']:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, '**', f'*.{ext}'), recursive=True))
        self.image_paths.sort()  # Sort for reproducibility

    def load_image(self, path):
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((self.image_size, self.image_size))
            return img
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None

    def __len__(self):
        return len(self.image_paths)

    def image_to_patches(self, image):
        """
        Convert a PIL image into flattened patch vectors based on patch_mode.
        Returns a NumPy array of shape (num_patches, patch_vector_dim).
        """
        patches = []
        if self.patch_mode == 0:
            # Grid patches with separate height and width.
            for y in range(0, self.image_size, self.patch_height):
                for x in range(0, self.image_size, self.patch_width):
                    patch = image.crop((x, y, x + self.patch_width, y + self.patch_height))
                    patch = (np.array(patch).astype(np.float32) / 255.0) * 2 - 1
                    patches.append(patch.flatten())
        elif self.patch_mode == 1:
            # Rowwise: each patch is a horizontal strip with height = patch_height.
            for y in range(0, self.image_size, self.patch_height):
                patch = image.crop((0, y, self.image_size, y + self.patch_height))
                patch = (np.array(patch).astype(np.float32) / 255.0) * 2 - 1
                patches.append(patch.flatten())
        elif self.patch_mode == 2:
            # Columnwise: each patch is a vertical strip with width = patch_width.
            for x in range(0, self.image_size, self.patch_width):
                patch = image.crop((x, 0, x + self.patch_width, self.image_size))
                patch = (np.array(patch).astype(np.float32) / 255.0) * 2 - 1
                patches.append(patch.flatten())
        return np.stack(patches)

    def __getitem__(self, index):
        # Load and process image.
        path = self.image_paths[index]
        img = self.load_image(path)
        if img is None:
            # If loading fails, randomly select another sample.
            return self.__getitem__((index + 1) % len(self))
        patches = self.image_to_patches(img)  # Shape: (num_patches, patch_vector_dim)

        # Create a random start token (same as used in sampling).
        start_token = np.clip(np.random.normal(loc=0.0, scale=0.4, size=(1, self.patch_vector_dim)), -1, 1)# * 0.1

        # For teacher forcing: add noise to all patches except the last one.
        noise_scale = np.clip(np.random.normal(loc=0.0, scale=0.4, size=patches[:-1].shape), -1, 1)
        noisy_patches = patches[:-1] + np.random.randn(*patches[:-1].shape) * noise_scale

        # Input sequence: start token followed by noisy patches.
        input_seq = np.concatenate([start_token, noisy_patches], axis=0)
        target_seq = patches  # Target is the clean patch sequence.

        # Convert sequences to torch tensors.
        input_seq_tensor = torch.tensor(input_seq, dtype=torch.float32)
        target_seq_tensor = torch.tensor(target_seq, dtype=torch.float32)
        return input_seq_tensor, target_seq_tensor

#############################################
# Helper Functions for Image Processing
#############################################
def image_to_patches(image, patch_mode, patch_height, patch_width, image_size):
    """Convert a PIL image into flattened patch vectors."""
    patches = []
    if patch_mode == 0:
        for y in range(0, image_size, patch_height):
            for x in range(0, image_size, patch_width):
                patch = image.crop((x, y, x + patch_width, y + patch_height))
                patch = (np.array(patch).astype(np.float32) / 255.0) * 2 - 1
                patches.append(patch.flatten())
    elif patch_mode == 1:
        for y in range(0, image_size, patch_height):
            patch = image.crop((0, y, image_size, y+patch_height))
            patch = (np.array(patch).astype(np.float32) / 255.0) * 2 - 1
            patches.append(patch.flatten())
    elif patch_mode == 2:
        for x in range(0, image_size, patch_width):
            patch = image.crop((x, 0, x+patch_width, image_size))
            patch = (np.array(patch).astype(np.float32) / 255.0) * 2 - 1
            patches.append(patch.flatten())
    return np.stack(patches)

def patches_to_image(patches, patch_mode, patch_height, patch_width, image_size):
    """
    Convert a sequence of patch vectors back into an image.
    For mode 0: patches are arranged in a grid with patch dimensions patch_height x patch_width.
    For mode 1: each patch is a horizontal strip with height = patch_height.
    For mode 2: each patch is a vertical strip with width = patch_width.
    """
    if patch_mode == 0:
        num_patches_y = image_size // patch_height
        num_patches_x = image_size // patch_width
        patches = patches.reshape(num_patches_y, num_patches_x, patch_height, patch_width, 3)
        patches = patches.transpose(0, 2, 1, 3, 4)
        image = (patches.reshape(image_size, image_size, 3) + 1) / 2
        return np.clip(image, 0, 1)
    elif patch_mode == 1:
        # Rowwise: each patch is (patch_height, image_size, 3)
        rows = []
        for patch in patches:
            row = patch.reshape(patch_height, image_size, 3)
            rows.append(row)
        image = np.concatenate(rows, axis=0)
        image = (image + 1) / 2
        return np.clip(image, 0, 1)
    elif patch_mode == 2:
        # Columnwise: each patch is (image_size, patch_width, 3)
        cols = []
        for patch in patches:
            col = patch.reshape(image_size, patch_width, 3)
            cols.append(col)
        image = np.concatenate(cols, axis=1)
        image = (image + 1) / 2
        return np.clip(image, 0, 1)

#############################################
# Autoregressive Generation Function
#############################################
def generate_image(model, config, device):
    """
    Generate a full image by autoregressively predicting patches.
    An extra START OF SEQUENCE token is prepended (but later dropped)
    so that the final image does not include the random seed patch.
    Noise is added at each step using a Gaussian multiplier.
    """
    image_size = config['image_size']
    patch_mode = config['patch_mode']
    if patch_mode == 0:
        patch_height = config['patch_height']
        patch_width = config['patch_width']
        patch_vector_dim = patch_height * patch_width * 3
        num_patches = (image_size // patch_height) * (image_size // patch_width)
    elif patch_mode == 1:
        patch_height = config['patch_height']
        patch_vector_dim = image_size * patch_height * 3
        num_patches = (image_size // patch_height)
    elif patch_mode == 2:
        patch_width = config['patch_width']
        patch_vector_dim = image_size * patch_width * 3
        num_patches = (image_size // patch_width)

    model_type = config['model_type']
    generated_patches = []
    model.eval()
    with torch.no_grad():
        # Helper: add Gaussian noise.
        def add_noise(tensor):
            noise_scale = torch.clamp(torch.normal(mean=0.0, std=0.4, size=tensor.shape, device=device), -1, 1)
            return tensor + (torch.randn_like(tensor) * noise_scale)

        if model_type in ["1", "2", "3", "6"]:
            start_token = torch.clamp(torch.normal(mean=0.0, std=0.4, size=(1, 1, patch_vector_dim), device=device), -1, 1)
            #torch.randn(1, 1, patch_vector_dim).to(device)# * 0.1
            current_patch = start_token
            hidden = None
            prev_hiddens = None
            for i in range(num_patches):
                if model_type not in ["3"]:
                    res = model(current_patch, hidden)
                else:
                    res, prev_hiddens = model(current_patch, prev_hiddens, True)
                if isinstance(res, tuple):
                    output, hidden = res
                else:
                    output = res
                sampled_patch = add_noise(output)
                generated_patches.append(output.squeeze(0).squeeze(0).cpu().numpy())
                current_patch = sampled_patch
        elif model_type in ["4", "5", "7", "8", "9", "10", "11", "12"]:
            start_token = torch.randn(1, 1, patch_vector_dim).to(device) * 0.1
            current_seq = start_token  # shape: (1, 1, patch_vector_dim)
            for i in range(num_patches):
                output = model(current_seq)
                next_patch = output[:, -1:, :]  # Use the last token as next prediction.
                next_patch_noisy = add_noise(next_patch)
                generated_patch = next_patch.squeeze(0).squeeze(0).cpu().numpy()
                generated_patches.append(generated_patch)
                current_seq = torch.cat([current_seq, next_patch_noisy], dim=1)
        else:
            raise ValueError("Invalid model type in configuration.")

    generated_patches = np.array(generated_patches)
    # Reconstruct the image using the updated patches_to_image function.
    if patch_mode == 0:
        img = patches_to_image(generated_patches, patch_mode, config['patch_height'], config['patch_width'], image_size)
    elif patch_mode == 1:
        img = patches_to_image(generated_patches, patch_mode, config['patch_height'], None, image_size)
    elif patch_mode == 2:
        img = patches_to_image(generated_patches, patch_mode, None, config['patch_width'], image_size)
    if config.get("denoise", False):
        img = denoise_image(img)
    return img

def save_sample_image(image, filename):
    """ Save a generated image to disk. """
    img = Image.fromarray((image * 255).astype(np.uint8))
    img.save(filename)
    print(f"Saved sample image to {filename}")

#############################################
# Helper Functions for minGRU and Others
#############################################

def exists(v):
    return v is not None

@script
def heinsen_associative_scan_log(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(dim=1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()

@script
def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())

@script
def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

#############################################
# Model Definitions
#############################################
class LRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_f = nn.Linear(input_size, hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        #self.b_f = nn.Parameter(torch.zeros(hidden_size))
        self.s_s = nn.Parameter(torch.ones(hidden_size))
        self.once = 0
        
        # Replace PReLU with LEAF activation
        self.act1 = GoLU()#torch.tanh#ParabolicConeActivation(hidden_size)#(1)
        self.act2 = GoLU()#F.sigmoid#ParabolicConeActivation(hidden_size)#(1)
        
        # Set self.norm based on layerindex
        #self.norm = 0.5 ** layerindex


    def forward(self, x, h_prev):
        # Compute candidate hidden state
        h_tilde = self.act1(self.norm1(self.W_h(x)))
        
        # Compute forget gate
        f_t = self.act2(self.norm2(self.U_f(h_prev) + self.W_f(x)))
        
        # Update hidden state
        h_t = (self.s_s - f_t) * h_prev + f_t * h_tilde
        
        return self.norm3(h_t)# * self.norm
# GLU using TeLU activation
class GLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GLU, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
        self.act = TeLU()
    def forward(self, x):
        return self.linear(x) * self.act(self.gate(x))
class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * (self.gamma + 1)
# minGRU implementation (single layer)
# Revised minGRU implementation (single layer)
class minGRU(nn.Module):
    def __init__(self, dim, expansion_factor=4.):
        super().__init__()
        dim_inner = int(dim * expansion_factor)
        self.norm = RMSNorm(dim)
        self.to_hidden_and_gate = nn.Linear(dim, dim_inner * 2)
        self.to_out = nn.Linear(dim_inner, dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        """
        x: tensor of shape (B, T, D)
        prev_hidden: tensor of shape (B, 1, D) representing the start token hidden state (if available)
        Returns output of shape (B, T, D) where T matches the input length (i.e. including the start token).
        """
        T = x.shape[1]
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim=-1)
        
        # For a single token input, use the simple branch
        if T == 1:
            hidden = g(hidden)
            gate = gate.sigmoid()
            out = torch.lerp(prev_hidden, hidden, gate) if exists(prev_hidden) else (hidden * gate)
            next_hidden = out[:, -1:]
            out = self.to_out(out)
            if not return_next_prev_hidden:
                return out
            return out, next_hidden

        # For multi-token inputs, perform an associative scan.
        log_coeffs = -F.softplus(gate)    # (B, T, D)
        log_z = -F.softplus(-gate)         # (B, T, D)
        log_tilde_h = log_g(hidden)        # (B, T, D)
        log_values = log_z + log_tilde_h   # (B, T, D)
        
        # Pad both log_coeffs and log_values with an extra token at the beginning.
        log_coeffs_padded = F.pad(log_coeffs, (0, 0, 1, 0))  # (B, T+1, D)
        log_values_padded = F.pad(log_values, (0, 0, 1, 0))  # (B, T+1, D)
        
        # If a previous hidden state exists, use it for the first token.
        if exists(prev_hidden):
            init_log_value = torch.log(prev_hidden)  # (B, 1, D)
            log_values_padded[:, 0:1, :] = init_log_value

        # Run the associative scan.
        out_scan = heinsen_associative_scan_log(log_coeffs_padded, log_values_padded)  # (B, T+1, D)
        # Remove the extra initial token to match the input sequence length.
        out = out_scan[:, 1:, :]  # (B, T, D)
        next_hidden = out[:, -1:, :]

        out = self.to_out(out)
        if not return_next_prev_hidden:
            return out
        return (out, next_hidden)

class ParabolicConeActivation(nn.Module):
    def __init__(self, num_features, is_conv=False):
        super(ParabolicConeActivation, self).__init__()
        shape = (1, num_features, 1, 1) if is_conv else (num_features,)
        self.beta = nn.Parameter(torch.full(shape, 2.0))
        self.alpha = nn.Parameter(torch.full(shape, 1.0))
        self.gamma = nn.Parameter(torch.full(shape, 1.0))
        self.sigma = nn.Parameter(torch.full(shape, 1.0))

    def forward(self, x):
        x = x * self.sigma
        return self.gamma * (self.alpha + (x * (self.beta - x)))

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        #nn.Linear(dim, dim_inner),
        ParabolicConeActivation(dim),#SRSFunction(1, 1., 1.),
        #nn.Linear(dim_inner, dim)
    )


class MultiLayerMinGRU(nn.Module):
    """
    A multi-layer MinGRU adapted for continuous inputs.
    
    This module stacks several layers. Each layer consists of:
      - (Optional) A causal depth-wise convolution.
      - RMSNorm.
      - A minGRU block with a residual connection.
      - A second RMSNorm.
      - A feedforward block (here, a lightweight activation module)
        with its own residual connection.
    
    Parameters:
        dim (int): The dimensionality of the input and hidden states.
        num_layers (int): The number of minGRU layers.
        expansion_factor (float): Expansion factor used in each minGRU.
        ff_mult (float): Feedforward multiplier (currently used only to choose the feedforward activation scale).
        conv_kernel_size (int): Kernel size for the optional causal depth-wise convolution.
        enable_conv (bool): Whether to include the convolutional block.
    """
    def __init__(
        self,
        dim,
        hidden_dim,
        num_layers=1,
        expansion_factor=1.5,
        ff_mult=4,
        conv_kernel_size=3,
        enable_conv=False
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.proj = nn.Linear(dim, hidden_dim) if dim != hidden_dim else nn.Identity()
        self.final_layer = nn.Linear(hidden_dim, dim) if dim != hidden_dim else nn.Identity()
        
        for _ in range(num_layers):
            layer_dict = {}
            # Optional convolution block
            if enable_conv:
                layer_dict["conv"] = CausalDepthWiseConv1d(dim, conv_kernel_size)
            else:
                layer_dict["conv"] = None
            # First normalization before the minGRU
            layer_dict["norm1"] = RMSNorm(hidden_dim)
            # The minGRU layer (assumed to be imported from minGRU_pytorch.minGRU)
            layer_dict["minGRU"] = minGRU(hidden_dim, expansion_factor=expansion_factor)
            # Second normalization before the feedforward
            layer_dict["norm2"] = RMSNorm(hidden_dim)
            layer_dict["parbol"] = ParabolicConeActivation(hidden_dim)
            # FeedForward block: here we use a simple activation inspired by the original design.
            # You can extend this with a more complex MLP if needed.
            layer_dict["ff"] = FeedForward(hidden_dim, mult=ff_mult)
            
            # Wrap the components in an nn.ModuleDict for easier handling.
            self.layers.append(nn.ModuleDict(layer_dict))
            
        # A final normalization on the output.
        self.final_norm = RMSNorm(hidden_dim)
    
    def forward(self, x, prev_hiddens=None, return_next_hidden=False):
        """
        Forward pass for continuous input x.
        
        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, dim).
            prev_hiddens (list or None): A list containing the previous hidden state for each layer.
                If None, each layer starts with no previous hidden state.
            return_next_hidden (bool): If True, the method returns a tuple (output, new_hidden_states).
        
        Returns:
            Tensor: Processed output of shape (batch, sequence_length, dim), and optionally the new hidden states.
        """
        # If no previous hidden states are provided, initialize a list of Nones.
        if prev_hiddens is None:
            prev_hiddens = [None] * self.num_layers
        
        new_hiddens = []
        x = self.proj(x)
        for i, layer in enumerate(self.layers):
            # Optional convolution with residual connection.
            if layer["conv"] is not None:
                x = layer["conv"](x) + x

            # Normalize input for the minGRU.
            x_norm = layer["norm1"](x)
            # Process with the minGRU.
            result = layer["minGRU"](x_norm, prev_hidden=prev_hiddens[i], return_next_prev_hidden=return_next_hidden)
            if isinstance(result, tuple):
                gru_out, new_hidden = result
            else:
                gru_out = result
                new_hidden = None
            # Residual connection after minGRU.
            x = gru_out + x
            new_hiddens.append(new_hidden)
            
            # Second normalization before feedforward.
            x_norm2 = layer["norm2"](x)
            # Apply feedforward activation.
            ff_out = x_norm2#layer["ff"](x_norm2)
            # Residual connection after feedforward.
            x = ff_out + x

        x = self.final_norm(x)
        x = self.final_layer(x)
        if return_next_hidden:
            return x, new_hiddens
        return x



# LSTM model
class PatchLSTM(nn.Module):
    def __init__(self, patch_vector_dim, lstm_dim, lstm_layers):
        super(PatchLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=patch_vector_dim, hidden_size=lstm_dim, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(lstm_dim, patch_vector_dim)
    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        return self.fc(out), hidden

# GRU model
import torch
import torch.nn as nn

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Update gate parameters
        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)
        # Reset gate parameters
        self.W_r = nn.Linear(input_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        # Candidate hidden state parameters
        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.sigmoid = TeLU() #torch.sigmoid
        self.tanh = GoLU() #torch.tanh
        self.normz = nn.LayerNorm(hidden_size)
        self.normr = nn.LayerNorm(hidden_size)

    def forward(self, x, h_prev):
        z = torch.sigmoid(self.normz(self.W_z(x) + self.U_z(h_prev)))
        r = torch.sigmoid(self.normr(self.W_r(x) + self.U_r(h_prev)))
        h_tilde = torch.tanh(self.W_h(x) + self.U_h(r * h_prev))
        h_new = (1 - z) * h_prev + z * h_tilde
        return h_new

class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(CustomGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # Create a ModuleList of GRUCells for each layer.
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.cells.append(GRUCell(layer_input_size, hidden_size))
    
    def forward(self, x, h0=None):
        """
        x: Tensor of shape (batch, seq_len, input_size)
        h0: Optional initial hidden state of shape (num_layers, batch, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        # Initialize hidden state if not provided.
        if h0 is None:
            h = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        else:
            # Split h0 into a list for each layer.
            h = [h0[i] for i in range(self.num_layers)]
        
        outputs = []
        # Process each time step.
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_new = []
            for layer in range(self.num_layers):
                cell = self.cells[layer]
                h_prev = h[layer]
                h_current = cell(x_t, h_prev)
                h_new.append(h_current)
                # The output of the current layer is the input for the next.
                x_t = h_current
            h = h_new
            # Save the output of the last layer.
            outputs.append(h[-1].unsqueeze(1))
        
        # Concatenate outputs along the time dimension.
        output_seq = torch.cat(outputs, dim=1)  # shape: (batch, seq_len, hidden_size)
        # Stack the final hidden states from all layers.
        h_final = torch.stack(h, dim=0)  # shape: (num_layers, batch, hidden_size)
        return output_seq, h_final

class PatchGRU(nn.Module):
    def __init__(self, patch_vector_dim, gru_dim, gru_layers):
        super(PatchGRU, self).__init__()
        # Replace nn.GRU with our CustomGRU.
        self.gru = nn.GRU(input_size=patch_vector_dim, hidden_size=gru_dim, num_layers=gru_layers)
        self.fc = nn.Linear(gru_dim, patch_vector_dim)
    
    def forward(self, x, hidden=None):
        out, hidden = self.gru(x, hidden)
        return self.fc(out), hidden


# MLP model using TeLU activation and GPT-2 style positional embeddings
class PatchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        # Learned positional embeddings (GPT-2 style)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim)] +
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(layers - 1)] +
            [nn.Linear(hidden_dim, input_dim)]
        )
        self.activation = GoLU()
    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len, input_dim)
        seq_len = x.shape[1]
        pos_emb = self.pos_embedding[:, :seq_len, :]
        x = self.proj(x)
        x = x + pos_emb
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

# Transformer (GPT-2 decoder only) model



import torch
import torch.nn as nn

# Define a transformer block that includes a causal self-attention layer and an MLP.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming RMSNorm and GoLU are defined somewhere else in your code.
# For this example, I'll assume they work similarly to LayerNorm and GELU.

class GPT2TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_hidden_dim=None):
        super().__init__()
        self.ln1 = RMSNorm(hidden_dim)
        self.ln2 = RMSNorm(hidden_dim)
        # MultiheadAttention expects inputs with shape (seq_len, batch, hidden_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        # Set mlp_hidden_dim (typically 4x the hidden dimension)
        mlp_hidden_dim = mlp_hidden_dim or hidden_dim * 4
        self.gamma = nn.Parameter(torch.Tensor([0.0]))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            GoLU(),  # using a GELU-style activation (or TeLU as specified)
            nn.Linear(mlp_hidden_dim, hidden_dim)
        )

    def forward(self, x, attn_mask=None):
        # x: (batch, seq_len, hidden_dim)
        residual = x
        x_norm = self.ln1(x)
        # Transpose for multihead attention: (seq_len, batch, hidden_dim)
        x_norm = x_norm.transpose(0, 1)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        # Bring back to (batch, seq_len, hidden_dim)
        attn_out = attn_out.transpose(0, 1)
        x = residual + attn_out

        # Apply second layer norm, the feed-forward MLP, and add residual.
        x = x + self.mlp(self.ln2(x))
        return x

# Define a GPT-2 style transformer model adapted for continuous image patches.
class PatchTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, num_heads, max_seq_len):
        """
        Args:
            input_dim (int): Dimension of the continuous input (e.g. image patch size).
            hidden_dim (int): Transformer hidden dimension.
            layers (int): Number of transformer blocks.
            num_heads (int): Number of attention heads.
            max_seq_len (int): Maximum sequence length.
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        
        # Project input patches into the transformer hidden space.
        self.token_embedding = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        # Learned positional embeddings in the hidden dimension.
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))
        
        # Create a stack of transformer blocks.
        self.blocks = nn.ModuleList([
            GPT2TransformerBlock(hidden_dim, num_heads) for _ in range(layers)
        ])
        
        # Final normalization and projection back to the continuous output dimension.
        self.final_layer_norm = nn.LayerNorm(hidden_dim)
        self.final_linear = nn.Linear(hidden_dim, input_dim)  # regression head for continuous outputs
        
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Tensor: Output tensor of shape (batch, seq_len, input_dim)
        """
        batch, seq_len, _ = x.shape
        
        # Project input tokens into the hidden space.
        x = self.token_embedding(x)  # shape: (batch, seq_len, hidden_dim)
        
        # Add learned positional embeddings.
        pos_emb = self.pos_embedding[:, :seq_len, :]
        x = x + pos_emb
        
        # Create a causal mask to enforce autoregressive behavior.
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1).to(x.device)
        
        # Pass through each transformer block.
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)
        
        # Final layer normalization and projection.
        x = self.final_layer_norm(x)
        x = self.final_linear(x)
        return x


import torch
import torch.nn as nn

# Optional: Define an alternative activation if desired.
# Here we simply use the standard GELU by default.
class GoLU(nn.Module):
    def forward(self, x):
        return torch.nn.functional.gelu(x)

# Rotary Multihead Attention (a drop-in replacement for nn.MultiheadAttention)
class RotaryMultiheadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        # x: (seq_len, batch, hidden_dim)
        seq_len, batch, _ = x.size()
        qkv = self.qkv_proj(x)  # shape: (seq_len, batch, 3*hidden_dim)
        qkv = qkv.reshape(seq_len, batch, self.num_heads, 3 * self.head_dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # each: (seq_len, batch, num_heads, head_dim)
        # Permute to (batch, num_heads, seq_len, head_dim)
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)
        # Compute rotary embeddings (sinusoidal)
        cos, sin = self.get_rotary_embeddings(seq_len, self.head_dim, x.device)
        q = self.apply_rotary(q, cos, sin)
        k = self.apply_rotary(k, cos, sin)
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            # attn_mask should broadcast to (batch, num_heads, seq_len, seq_len)
            attn_scores = attn_scores + attn_mask.unsqueeze(0).unsqueeze(0)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)
        # Reassemble output: (seq_len, batch, hidden_dim)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()
        attn_output = attn_output.view(seq_len, batch, self.hidden_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output

    def get_rotary_embeddings(self, seq_len, head_dim, device):
        # Compute rotary embeddings following common practice.
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, head_dim/2)
        # Duplicate so that the result has dimension head_dim.
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        cos = emb.cos()[None, None, :, :]  # (1, 1, seq_len, head_dim)
        sin = emb.sin()[None, None, :, :]  # (1, 1, seq_len, head_dim)
        return cos, sin

    def apply_rotary(self, x, cos, sin):
        # x: (batch, num_heads, seq_len, head_dim)
        # Rotate half of the dimensions.
        x1, x2 = x.chunk(2, dim=-1)
        x_rot = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (x_rot * sin)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Stochastic Depth implementation.
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        # Generate binary tensor mask.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = (1 - self.drop_prob) + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(1 - self.drop_prob) * random_tensor

# Modern Transformer Block with dropout, layer scaling, and optional rotary attention.
class ModernTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_hidden_dim=None,
                 attn_dropout=0.1, mlp_dropout=0.1, layer_scale_init_value=1e-5,
                 use_rotary=False, activation=nn.GELU, drop_path_rate=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.use_rotary = use_rotary
        if use_rotary:
            # Assumes a custom rotary attention module is defined elsewhere.
            self.self_attn = RotaryMultiheadAttention(hidden_dim, num_heads, dropout=attn_dropout)
        else:
            self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=attn_dropout)
        mlp_hidden_dim = mlp_hidden_dim or hidden_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            activation(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(mlp_dropout)
        )
        # Learnable scaling for each residual branch.
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(hidden_dim))
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(hidden_dim))
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.drop_path = DropPath(drop_path_rate)
    
    def forward(self, x, attn_mask=None):
        # x: (batch, seq_len, hidden_dim)
        residual = x
        x_norm = self.ln1(x)
        # Transpose for attention (seq_len, batch, hidden_dim).
        x_norm = x_norm.transpose(0, 1)
        if self.use_rotary:
            attn_out = self.self_attn(x_norm, attn_mask=attn_mask)
        else:
            attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        attn_out = attn_out.transpose(0, 1)  # back to (batch, seq_len, hidden_dim)
        # Apply layer scaling, dropout, and drop path to the attention branch.
        x = residual + self.drop_path(self.attn_dropout(self.layer_scale_1 * attn_out))
        # MLP branch with residual connection and drop path.
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.ln2(x)))
        return x

# GPT-2 style Transformer for image patches with modern improvements.
class ModernPatchTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, num_heads, max_seq_len,
                 attn_dropout=0.1, mlp_dropout=0.1, layer_scale_init_value=1e-5,
                 use_rotary=True, activation=nn.GELU, drop_path_rate=0.0):
        """
        Args:
            input_dim (int): Dimension of the input image patches.
            hidden_dim (int): Transformer hidden dimension.
            layers (int): Number of transformer blocks.
            num_heads (int): Number of attention heads.
            max_seq_len (int): Maximum sequence length.
            attn_dropout (float): Dropout rate for attention weights.
            mlp_dropout (float): Dropout rate for the MLP.
            layer_scale_init_value (float): Initial value for layer scaling parameters.
            use_rotary (bool): If True, use rotary positional embeddings in attention.
            activation (nn.Module): Activation function for the MLP.
            drop_path_rate (float): Overall stochastic depth rate.
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        # When using rotary embeddings, absolute positional embeddings are often omitted.
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim)) if not use_rotary else None
        # Optionally project input tokens to the hidden dimension.
        self.token_embedding = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        # Schedule drop path rates across layers (if more than one layer).
        dpr = [drop_path_rate * float(i) / (layers - 1) for i in range(layers)] if layers > 1 else [0.0]
        self.blocks = nn.ModuleList([
            ModernTransformerBlock(hidden_dim, num_heads,
                                     attn_dropout=attn_dropout,
                                     mlp_dropout=mlp_dropout,
                                     layer_scale_init_value=layer_scale_init_value,
                                     use_rotary=use_rotary,
                                     activation=activation,
                                     drop_path_rate=dpr[i])
            for i in range(layers)
        ])
        self.final_layer_norm = nn.LayerNorm(hidden_dim)
        self.final_linear = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, input_dim)
        Returns:
            Tensor: Output tensor of shape (batch, seq_len, input_dim)
        """
        batch, seq_len, _ = x.shape
        x = self.token_embedding(x)
        if self.pos_embedding is not None:
            pos_emb = self.pos_embedding[:, :seq_len, :]
            x = x + pos_emb
        # Create a causal mask (upper triangle with -inf) for autoregressive behavior.
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1).to(x.device)
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)
        x = self.final_layer_norm(x)
        x = self.final_linear(x)
        return x


from transformers import RwkvConfig, RwkvForCausalLM
import torch.nn as nn
import math
import torch
import torch.nn as nn

# Assume GoLU is defined elsewhere. For instance:
# class GoLU(nn.Module):
#     def forward(self, x):
#         return torch.nn.functional.gelu(x)

import math
import torch
import torch.nn as nn

# Assume GoLU is defined elsewhere. For instance:
# class GoLU(nn.Module):
#     def forward(self, x):
#         return torch.nn.functional.gelu(x)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_hidden_dim=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        # MultiheadAttention expects inputs with shape (seq_len, batch, hidden_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        # Set mlp_hidden_dim (typically 4x the hidden dimension)
        mlp_hidden_dim = mlp_hidden_dim or hidden_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            GoLU(),  # using TeLU activation as specified
            nn.Linear(mlp_hidden_dim, hidden_dim)
        )

    def forward(self, x, attn_mask=None):
        # x: (batch, seq_len, hidden_dim)
        residual = x
        x_norm = self.ln1(x)
        # Transpose for multihead attention: (seq_len, batch, hidden_dim)
        x_norm = x_norm.transpose(0, 1)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        # Bring back to (batch, seq_len, hidden_dim)
        attn_out = attn_out.transpose(0, 1)
        x = residual + attn_out

        # Apply second layer norm, the feed-forward MLP, and add residual.
        x = x + self.mlp(self.ln2(x))
        return x

class PatchAliBi(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, num_heads, max_seq_len):
        """
        Args:
            input_dim (int): Dimension of the input image patches.
            hidden_dim (int): Transformer hidden dimension.
            layers (int): Number of transformer blocks.
            num_heads (int): Number of attention heads.
            max_seq_len (int): Maximum sequence length (i.e. maximum number of image patches).
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        
        # Instead of learned positional embeddings, we only use a token embedding.
        self.token_embedding = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        # Create a stack of transformer blocks.
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads) for _ in range(layers)
        ])
        
        # Final normalization and projection back to the input dimension.
        self.final_layer_norm = nn.LayerNorm(hidden_dim)
        self.final_linear = nn.Linear(hidden_dim, input_dim)
        
        # Precompute and register AliBi slopes (one per attention head)
        self.register_buffer("slopes", self.get_slopes(num_heads))
        
    @staticmethod
    def get_slopes(n):
        """
        Compute slopes for the AliBi mechanism.
        The following implementation follows the method described in the AliBi paper.
        """
        def get_slopes_power_of_2(n):
            # Compute slopes for when n is a power of 2.
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            return [start * (start ** i) for i in range(n)]
        
        if math.log2(n).is_integer():
            slopes = get_slopes_power_of_2(n)
        else:
            # For non-powers of 2, get slopes for the closest lower power of 2
            closest_power = 2 ** math.floor(math.log2(n))
            slopes = get_slopes_power_of_2(closest_power)
            # And then extend with additional slopes computed recursively.
            extra_slopes = PatchAliBi.get_slopes(2 * closest_power)[0::2]
            slopes.extend(extra_slopes[: n - closest_power])
        return torch.tensor(slopes, dtype=torch.float32)
        
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Tensor: Output tensor of shape (batch, seq_len, input_dim)
        """
        batch, seq_len, _ = x.shape
        
        # Project tokens to hidden_dim if needed.
        x = self.token_embedding(x)
        
        # --- Compute AliBi Mask ---
        device = x.device
        seq_range = torch.arange(seq_len, device=device)
        # Compute relative positions (i - j)
        relative_positions = seq_range.unsqueeze(1) - seq_range.unsqueeze(0)  # Shape: (seq_len, seq_len)
        
        # Causal mask: positions where i < j get -inf.
        causal_mask = torch.where(
            relative_positions < 0,
            torch.tensor(float('-inf'), device=device),
            torch.tensor(0.0, device=device)
        )
        # Clamp relative positions for AliBi bias calculation (only non-negative differences)
        relative_positions_clamped = torch.clamp(relative_positions, min=0)
        # Reshape slopes to (num_heads, 1, 1)
        slopes = self.slopes.view(-1, 1, 1)
        # Compute per-head bias: bias = -slope * (i - j)
        aliBi_bias = - slopes * relative_positions_clamped  # Shape: (num_heads, seq_len, seq_len)
        # Combine the AliBi bias with the causal mask (unsqueeze causal mask to add a head dimension)
        final_mask = aliBi_bias + causal_mask.unsqueeze(0)  # Shape: (num_heads, seq_len, seq_len)
        
        # Expand the mask for the batch dimension.
        # First, add a batch dimension: shape (1, num_heads, seq_len, seq_len)
        # Then, expand it to (batch, num_heads, seq_len, seq_len)
        # Finally, reshape to (batch*num_heads, seq_len, seq_len) as required by nn.MultiheadAttention.
        final_mask = final_mask.unsqueeze(0).expand(batch, -1, seq_len, seq_len)
        final_mask = final_mask.reshape(batch * self.num_heads, seq_len, seq_len)
        
        # --- Pass through Transformer Blocks ---
        for block in self.blocks:
            x = block(x, attn_mask=final_mask)
        
        # Final normalization and projection.
        x = self.final_layer_norm(x)
        x = self.final_linear(x)
        return x


class PatchRWKV(nn.Module):
    def __init__(self, patch_vector_dim, layers, max_len):
        super().__init__()
        # Set vocab_size equal to patch_vector_dim so that the output shape is correct.
        config = RwkvConfig(
            vocab_size=patch_vector_dim,
            context_length=max_len,
            hidden_size=patch_vector_dim,
            num_hidden_layers=layers,
            torch_dtype=torch.float32,
            # Other RWKV parameters will take their default values.
        )
        self.model = RwkvForCausalLM(config)

    def forward(self, x, hidden=None):
        # Use inputs_embeds to pass our continuous tokens.
        return self.model(inputs_embeds=x).logits

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Lightweight Transformer using a minimal TransformerDecoder.
class PatchLiteTransformer(nn.Module):
    def __init__(self, patch_vector_dim, layers, max_len):
        """
        A lightweight transformer that uses a single-head attention mechanism and a minimal feedforward network.
        Designed to work with patch tokens.
        """
        super().__init__()
        self.patch_vector_dim = patch_vector_dim
        self.max_len = max_len
        
        # Linear embedding to project input tokens (patch vectors) into the transformer space.
        self.embedding = nn.Linear(patch_vector_dim, patch_vector_dim)
        
        # Learned positional embeddings (like in GPT-2).
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, patch_vector_dim))
        
        # Create a TransformerDecoder layer with a single attention head.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=patch_vector_dim, 
            nhead=1, 
            dim_feedforward=patch_vector_dim * 2  # minimal feedforward size
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        
        # Final linear projection back to patch_vector_dim.
        self.output_linear = nn.Linear(patch_vector_dim, patch_vector_dim)

    def forward(self, x, hidden=None):
        """
        x: Tensor of shape (batch, seq_len, patch_vector_dim)
        Returns tensor of the same shape.
        """
        batch_size, seq_len, _ = x.shape
        
        # Project the input.
        x = self.embedding(x)
        # Add positional embeddings.
        pos_emb = self.pos_embedding[:, :seq_len, :]
        x = x + pos_emb
        
        # Transformer modules expect (seq_len, batch, embedding_dim).
        x = x.transpose(0, 1)
        
        # Create an autoregressive (square subsequent) mask.
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Use the transformer decoder. Here we simply use x as both the target and the memory.
        x = self.transformer_decoder(tgt=x, memory=x, tgt_mask=tgt_mask)
        
        # Bring back to (batch, seq_len, embedding_dim) and project the output.
        x = x.transpose(0, 1)
        x = self.output_linear(x)
        return x

# Residual form of PatchMLP with ReZero connections.
import torch
import torch.nn as nn

# Assuming TeLU is defined elsewhere. If not, you could use nn.Tanh() or any other activation.

import torch
import torch.nn as nn

# DropPath (Stochastic Depth) implementation.
class DropaPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        # Generate binary mask.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = (1 - self.drop_prob) + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize.
        return x.div(1 - self.drop_prob) * random_tensor

class ResidualPatchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, max_seq_len, drop_path_rate=0.0):
        """
        A PatchMLP that uses residual connections with DropPath regularization.
        Each residual branch is applied only if the layer's input and output dimensions match.
        Otherwise (e.g., when mapping from input_dim to hidden_dim or vice versa), the
        residual connection is omitted.
        
        Args:
            input_dim (int): Dimension of the input tokens.
            hidden_dim (int): Hidden dimension of the network.
            layers (int): Number of linear layers.
            max_seq_len (int): Maximum sequence length (for positional embeddings).
            drop_path_rate (float): Maximum drop path rate to apply across layers.
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        # Learned positional embeddings.
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))
        
        # Build a list of linear layers for the network.
        self.layers = nn.ModuleList()
        # Create a corresponding drop path module for each layer.
        self.drop_paths = nn.ModuleList()
        
        for i in range(layers):
            if i == 0:
                # First layer: map from input_dim to hidden_dim.
                layer = nn.Linear(hidden_dim, hidden_dim)
            elif i == layers - 1:
                # Last layer: map from hidden_dim back to input_dim.
                layer = nn.Linear(hidden_dim, input_dim)
            else:
                # Intermediate layers: maintain hidden dimension.
                layer = nn.Linear(hidden_dim, hidden_dim)
            self.layers.append(layer)
            # Schedule drop path rate if multiple layers are used.
            if layers > 1:
                layer_drop_rate = drop_path_rate * i / (layers - 1)
            else:
                layer_drop_rate = 0.0
            self.drop_paths.append(DropaPath(layer_drop_rate))
        
        # Activation function (assuming GoLU is defined elsewhere).
        self.activation = GoLU()

    def forward(self, x, hidden=None):
        """
        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, input_dim)
        Returns:
            Tensor: Output tensor of shape (batch, seq_len, output_dim)
        """
        seq_len = x.shape[1]
        # Add positional embeddings.
        pos_emb = self.pos_embedding[:, :seq_len, :]
        x = self.proj(x) + pos_emb
        
        # Apply each linear layer with a conditional residual connection.
        for i, layer in enumerate(self.layers):
            residual = x
            out = layer(x)
            # For intermediate layers, apply the activation.
            if i < len(self.layers) - 1:
                out = self.activation(out)
            # Apply the residual connection if the dimensions match.
            if residual.shape[-1] == out.shape[-1]:
                x = residual + self.drop_paths[i](out)
            else:
                x = out
        return x

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

class PatchDistilTransformer(nn.Module):
    def __init__(self, patch_vector_dim, layers, max_len):
        """
        A lightweight patch transformer based on a distilled GPT-2-like architecture.
        """
        super().__init__()
        config = GPT2Config(
            vocab_size=patch_vector_dim,
            n_positions=max_len,
            n_ctx=max_len,
            n_embd=patch_vector_dim,
            n_layer=layers,
            n_head=1,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            torch_dtype=torch.float32  # Set the dtype explicitly.
        )
        config.torch_dtype = torch.float32
        self.model = GPT2LMHeadModel(config)
    
    def forward(self, x, hidden=None):
        return self.model(inputs_embeds=x).logits

class PatchRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, max_seq_len, cell_type):
        """
        A patch-based RNN model with trainable positional embeddings.

        Args:
            input_dim (int): Dimension of the input patch vectors.
            hidden_dim (int): Hidden dimension for the RNN cells.
            layers (int): Number of RNN layers.
            max_seq_len (int): Maximum sequence length (including the start token).
            cell_type (nn.Module): A callable RNN cell class. It should be constructed as
                                   cell_type(in_dim, hidden_dim) and implement a forward
                                   method that accepts (input, hidden) and returns (output, new_hidden).
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        # Trainable positional embeddings, analogous to PatchMLP.
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, input_dim))
        self.layers = layers
        self.hidden_dim = hidden_dim

        # Create a stack of RNN cells. For the first layer, input dimension is input_dim;
        # for subsequent layers, it becomes hidden_dim.
        self.cells = nn.ModuleList()
        for i in range(layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(cell_type(in_dim, hidden_dim))
        
        # Final projection layer to map the last layer's output back to the original input_dim.
        self.output_linear = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, hidden=None):
        """
        Forward pass through the PatchRNN.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, input_dim).
            hidden (list or None): Optional list of hidden states for each layer.
                                   If None, each cell will initialize its hidden state.

        Returns:
            Tuple (output, new_hidden_states):
                - output (Tensor): Tensor of shape (batch, seq_len, input_dim).
                - new_hidden_states (list): List of updated hidden states for each RNN layer.
        """
        batch_size, seq_len, _ = x.shape
        # Add trainable positional embeddings to the input tokens.
        pos_emb = self.pos_embedding[:, :seq_len, :]
        x = x + pos_emb

        # Initialize hidden state list if none provided.
        if hidden is None:
            hidden = [None] * self.layers

        new_hidden_states = [None] * self.layers
        outputs = []
        # Process the sequence one time step at a time.
        # This loop is compatible with the RNN sampling method, where the sequence may be of length 1.
        for t in range(seq_len):
            input_t = x[:, t, :]  # (batch, input_dim) for the current time step.
            for layer_idx, cell in enumerate(self.cells):
                # Each cell is expected to return a tuple: (output, new_hidden)
                out_t, new_h = cell(input_t, hidden[layer_idx])
                input_t = out_t  # Output becomes input to the next layer.
                new_hidden_states[layer_idx] = new_h
            outputs.append(input_t)
            # Update hidden states for the next time step.
            hidden = new_hidden_states

        # Reassemble outputs into a tensor of shape (batch, seq_len, hidden_dim).
        out_seq = torch.stack(outputs, dim=1)
        # Project the last layer's output back to the original input dimension.
        out_seq = self.output_linear(out_seq)
        return out_seq, new_hidden_states

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchedGNNLayer(nn.Module):
    """
    A single graph attention layer for patches.
    For each node i (patch), the layer aggregates messages from nodes j where j < i (causal mask).
    The edge attention is computed from a small MLP on the concatenation of:
      - the features of node i,
      - the features of node j,
      - their difference (i - j).
    """
    def __init__(self, feature_dim):
        super(PatchedGNNLayer, self).__init__()
        self.feature_dim = feature_dim
        # The edge MLP: takes concatenated features of size 3 * feature_dim and outputs a single score.
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1)
        )
        # A linear projection for the "value" vectors.
        self.value_linear = nn.Linear(feature_dim, feature_dim)
        # Final projection for the aggregated message.
        self.out_linear = nn.Linear(feature_dim, feature_dim)
        # Layer normalization.
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, x):
        """
        x: Tensor of shape (batch, seq_len, feature_dim)
        Returns: Tensor of shape (batch, seq_len, feature_dim) after message aggregation.
        """
        batch, seq_len, dim = x.shape
        
        # Expand x so that we can compute pairwise combinations:
        # x_i: shape (batch, seq_len, seq_len, feature_dim)
        # x_j: shape (batch, seq_len, seq_len, feature_dim)
        x_i = x.unsqueeze(2).expand(-1, seq_len, seq_len, -1)
        x_j = x.unsqueeze(1).expand(-1, seq_len, seq_len, -1)
        
        # Concatenate features: [x_i, x_j, x_i - x_j]
        edge_input = torch.cat([x_i, x_j, x_i - x_j], dim=-1)  # (batch, seq_len, seq_len, 3*dim)
        
        # Compute edge scores and squeeze the last dimension: (batch, seq_len, seq_len)
        edge_scores = self.edge_mlp(edge_input).squeeze(-1)
        
        # Create a causal mask: only allow messages from nodes j where j < i.
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device), diagonal=-1)
        # Mask out scores where the mask is zero.
        edge_scores = edge_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Compute attention weights with softmax along the j dimension.
        attn_weights = F.softmax(edge_scores, dim=-1)  # (batch, seq_len, seq_len)
        
        # Compute values from the input nodes.
        values = self.value_linear(x)  # (batch, seq_len, dim)
        # For each node i, aggregate messages from nodes j using the computed attention weights.
        message = torch.bmm(attn_weights, values)  # (batch, seq_len, dim)
        
        # Project the aggregated message and add the residual connection.
        out = self.out_linear(message)
        out = x + out
        out = self.layer_norm(out)
        return out

class PatchedGNN(nn.Module):
    """
    The PatchedGNN model for autoregressive patch generation.
    It maps an input sequence of patch vectors to output patch vectors.
    """
    def __init__(self, patch_vector_dim, num_layers, hidden_dim=None):
        """
        Args:
            patch_vector_dim (int): Dimensionality of each patch vector.
            num_layers (int): Number of graph attention layers.
            hidden_dim (int): Internal hidden dimension. Defaults to patch_vector_dim.
        """
        super(PatchedGNN, self).__init__()
        self.hidden_dim = hidden_dim or patch_vector_dim
        # Project input patch vectors to the hidden dimension.
        self.input_linear = nn.Linear(patch_vector_dim, self.hidden_dim)
        # Stack several PatchedGNNLayer layers.
        self.layers = nn.ModuleList([PatchedGNNLayer(self.hidden_dim) for _ in range(num_layers)])
        # Project back to the original patch vector dimension.
        self.output_linear = nn.Linear(self.hidden_dim, patch_vector_dim)
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: Tensor of shape (batch, seq_len, patch_vector_dim)
        Returns:
            Tensor of shape (batch, seq_len, patch_vector_dim)
        """
        h = self.input_linear(x)
        for layer in self.layers:
            h = layer(h)
        out = self.output_linear(h)
        return out
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# SwiGLU Activation (gated MLP)
# -------------------------------
class SwiGLU(nn.Module):
    def forward(self, x):
        # Split the input along the last dimension into two halves
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b

# -------------------------------
# Selective SSM Layer
# -------------------------------
class SelectiveSSM(nn.Module):
    """
    Implements a selective state space model layer.
    
    For an input x of shape (B, L, D) (with D = model dimension),
    each channel d has its own SSM parameters (A, B, C) with state dimension N.
    
    The layer computes input-dependent effective discretization step Δ (delta_eff)
    via a softplus of a base bias plus a learned projection s_delta.
    
    Then for each time step t and each channel, we compute:
    
      A_bar = exp(Δ_t * A)
      B_bar = ((exp(Δ_t * A) - 1) / (A + eps)) * (B + s_B(x_t))
      
    and with an effective output projection
      C_eff = C + s_C(x_t)
    
    The recurrence (applied per channel independently) is:
    
      h_t = A_bar_t * h_{t-1} + B_bar_t * x_t   (elementwise, as A is assumed diagonal)
      y_t = sum_{n} [ C_eff_t[n] * h_t[n] ]
    
    (For numerical stability we add a small epsilon in the division.)
    """
    def __init__(self, dim, state_dim, eps=1e-6):
        super().__init__()
        self.dim = dim          # model (channel) dimension
        self.state_dim = state_dim  # SSM state dimension (N)
        self.eps = eps

        # Fixed (base) SSM parameters per channel.
        # We assume a diagonal structure so that for each channel d, A[d] is a vector of length state_dim.
        self.A = nn.Parameter(torch.randn(dim, state_dim) * 0.1)
        self.B = nn.Parameter(torch.randn(dim, state_dim) * 0.1)
        self.C = nn.Parameter(torch.randn(dim, state_dim) * 0.1)

        # Learnable bias for the discretization step Δ (per channel).
        self.delta_bias = nn.Parameter(torch.zeros(dim))
        # Projections for the selection mechanism.
        self.s_delta = nn.Linear(dim, dim)  # produces additional adjustment for Δ, shape (B, L, dim)
        # For s_B and s_C we want an output per channel and per state dimension.
        self.s_B = nn.Linear(dim, dim * state_dim)
        self.s_C = nn.Linear(dim, dim * state_dim)

    def forward(self, x):
        # x: (B, L, dim)
        B_size, L, D = x.shape  # D should equal self.dim
        device = x.device

        # Compute selection signals from input:
        s_delta = self.s_delta(x)            # shape (B, L, dim)
        # For s_B and s_C, reshape to (B, L, dim, state_dim)
        s_B = self.s_B(x).view(B_size, L, D, self.state_dim)
        s_C = self.s_C(x).view(B_size, L, D, self.state_dim)

        # Effective Δ per channel at each time: Δ_eff = softplus(delta_bias + s_delta)
        # delta_bias shape: (dim,) is broadcast to (B, L, dim)
        delta_eff = F.softplus(self.delta_bias + s_delta)  # (B, L, dim)

        # Compute effective (time-varying) parameters.
        # We compute A_bar = exp(delta_eff * A) for each channel.
        # Expand delta_eff to (B, L, dim, 1) and A to (1, 1, dim, state_dim)
        A_bar = torch.exp(delta_eff.unsqueeze(-1) * self.A.unsqueeze(0).unsqueeze(0))  # (B, L, dim, state_dim)
        
        # Effective B: add selection signal to fixed B.
        eff_B = self.B.unsqueeze(0).unsqueeze(0) + s_B  # (B, L, dim, state_dim)
        # Discretize B: B_bar = ((exp(delta_eff*A) - 1) / (A + eps)) * eff_B.
        B_bar = (A_bar - 1) / (self.A.unsqueeze(0).unsqueeze(0) + self.eps) * eff_B  # (B, L, dim, state_dim)

        # Effective C: add selection signal to fixed C.
        eff_C = self.C.unsqueeze(0).unsqueeze(0) + s_C  # (B, L, dim, state_dim)

        # Initialize hidden state h_0 = 0, per channel.
        h = torch.zeros(B_size, D, self.state_dim, device=device)
        outputs = []
        # Iterate over the sequence (time dimension).
        for t in range(L):
            # For time t, get parameters A_bar and B_bar for each channel:
            A_bar_t = A_bar[:, t, :, :]   # (B, dim, state_dim)
            B_bar_t = B_bar[:, t, :, :]   # (B, dim, state_dim)
            # x_t: (B, dim) -> unsqueeze last dim to (B, dim, 1)
            x_t = x[:, t].unsqueeze(-1)
            # Update recurrence per channel (elementwise multiplication; here A_bar_t acts like a gating factor):
            h = A_bar_t * h + B_bar_t * x_t  # (B, dim, state_dim)
            # Compute output y_t by projecting h with eff_C (elementwise product then sum over state dimension):
            y_t = (eff_C[:, t, :, :] * h).sum(dim=-1)  # (B, dim)
            outputs.append(y_t)
        # Stack over time → shape (B, L, dim)
        y = torch.stack(outputs, dim=1)
        return y

# -------------------------------
# Mamba Block
# -------------------------------
class MambaBlock(nn.Module):
    """
    A single Mamba block which combines:
      - A pre-SSM LayerNorm,
      - A selective SSM layer (the inner recurrence with selection),
      - A residual connection,
      - A post-SSM LayerNorm,
      - A gated MLP (implemented with SwiGLU) for further transformation,
      - Another residual connection.
    """
    def __init__(self, dim, state_dim, expansion_factor=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ssm = SelectiveSSM(dim, state_dim)
        self.norm2 = nn.LayerNorm(dim)
        # Gated MLP with expansion; note that for SwiGLU the hidden size is doubled.
        hidden_dim = dim * expansion_factor
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),  # output doubled for SwiGLU
            SwiGLU(),
            nn.Linear(hidden_dim, dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (B, L, dim)
        # SSM branch with residual connection.
        residual = x
        x_norm = self.norm1(x)
        ssm_out = self.ssm(x_norm)
        x = residual + self.dropout(ssm_out)
        
        # MLP branch with residual.
        residual = x
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = residual + self.dropout(mlp_out)
        return x

# -------------------------------
# PatchedMamba Model
# -------------------------------
class PatchedMamba(nn.Module):
    """
    A complete PatchedMamba model for autoregressive patch generation.
    
    It first projects the flattened patch vectors into a hidden space,
    then applies a stack of Mamba blocks (each with a selective SSM + gated MLP),
    and finally projects back to the patch vector dimension.
    """
    def __init__(self, patch_vector_dim, model_dim, num_layers, state_dim, expansion_factor=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(patch_vector_dim, model_dim)
        self.blocks = nn.ModuleList([
            MambaBlock(model_dim, state_dim, expansion_factor, dropout) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(model_dim, patch_vector_dim)
    
    def forward(self, x):
        # x: (B, L, patch_vector_dim)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x)
        return x


class CausalConv1d(nn.Module):
    """
    A 1D convolution layer with left-only padding so that the output at position t
    depends only on inputs at positions ≤ t.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        # Note: We set padding to 0 here and do manual left-padding in forward.
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=0, dilation=dilation, bias=bias)

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        # Compute required left padding: (kernel_size - 1) * dilation
        pad_amount = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (pad_amount, 0))  # pad only on the left side
        return self.conv(x)

# The new "patched" model using a masked CNN architecture.
class PatchMaskedCNN(nn.Module):
    """
    A model that autoregressively processes flattened image patch vectors via
    a stack of causal convolutional layers.
    
    Args:
        input_dim (int): Dimensionality of the flattened patch vector.
        hidden_dim (int): Dimension of the hidden representation (used in the CNN blocks).
        num_layers (int): Number of causal convolution blocks.
        kernel_size (int): Kernel size for the causal convolutions.
        dropout (float): Dropout rate applied after each convolution.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, kernel_size=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Project input to hidden_dim if needed.
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()
        
        # Build a stack of causal convolutional blocks.
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.Sequential(
                # The causal convolution operates along the sequence dimension.
                # We use a fixed dilation of 1 here (you could also experiment with varying dilations).
                CausalConv1d(hidden_dim, hidden_dim, kernel_size, dilation=1),
                GoLU(),
                nn.Dropout(dropout)
            )
            self.layers.append(block)
        
        # Layer normalization after the convolutional blocks.
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Project back to the input dimension if required.
        if hidden_dim != input_dim:
            self.output_proj = nn.Linear(hidden_dim, input_dim)
        else:
            self.output_proj = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_dim)
        
        Returns:
            Tensor of shape (batch, seq_len, input_dim)
        """
        # Project input tokens (patch vectors) into hidden space.
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        
        # Rearrange to (batch, channels, seq_len) for 1D convolution.
        x = x.transpose(1, 2)
        
        # Pass through each causal convolution block with residual connection.
        for layer in self.layers:
            residual = x
            x = layer(x)
            # Residual connection: add the block input to its output.
            x = x + residual
        
        # Rearrange back to (batch, seq_len, hidden_dim).
        x = x.transpose(1, 2)
        
        # Normalize the hidden representations.
        x = self.layer_norm(x)
        
        # Project back to the original patch vector dimension.
        x = self.output_proj(x)
        return x



#############################################
# Model Selection and Training
#############################################
def select_model(model_type, patch_vector_dim, hidden_dim, layers, max_seq_len=None, cell_type=None, attn_heads=8):
    """
    Select the architecture based on user input:
    "1" = LSTM, "2" = GRU, "3" = minGRU (multilayered), "4" = MLP (with positional embeddings), "5" = Transformer,
    "6" = RNN (will choose celltype after this), "7" = Residual MLP, "8" = AliBi Transformer, "9" = Modern Transformer.
    """
    if model_type == "1":
        return PatchLSTM(patch_vector_dim, hidden_dim, layers)
    elif model_type == "2":
        return PatchGRU(patch_vector_dim, hidden_dim, layers)
    elif model_type == "3":
        return MultiLayerMinGRU(patch_vector_dim, hidden_dim, num_layers=layers)
    elif model_type == "4":
        if max_seq_len is None:
            max_seq_len = 1024  # fallback default
        return PatchMLP(patch_vector_dim, hidden_dim, layers, max_seq_len)
    elif model_type == "5":
        return PatchTransformer(patch_vector_dim, hidden_dim, layers, attn_heads, max_seq_len)
    elif model_type == "6":
        return PatchRNN(patch_vector_dim, hidden_dim, layers, max_seq_len, cell_type)
    elif model_type == "7":
        if max_seq_len is None:
            max_seq_len = 1024  # fallback default
        return ResidualPatchMLP(patch_vector_dim, hidden_dim, layers, max_seq_len)
    elif model_type == "8":
        return PatchAliBi(patch_vector_dim, hidden_dim, layers, attn_heads, max_seq_len)
    elif model_type == "9":
        return ModernPatchTransformer(patch_vector_dim, hidden_dim, layers, attn_heads, max_seq_len)
    elif model_type == "10":
        return PatchedGNN(patch_vector_dim, layers, hidden_dim)
    elif model_type == "11":
        return PatchedMamba(patch_vector_dim, hidden_dim, layers, attn_heads)
    elif model_type == "12":
        return PatchMaskedCNN(patch_vector_dim, hidden_dim, layers)
    else:
        raise ValueError("Invalid model type selection.")
def train_model(config, cont=False):
    # Clean the sample folder.
    if not os.path.exists("LSTMimage"):
        os.makedirs("LSTMimage")
    else:
        for filename in os.listdir("LSTMimage"):
            file_path = os.path.join("LSTMimage", filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = config['image_size']
    patch_mode = config['patch_mode']

    if patch_mode == 0:
        patch_vector_dim = config['patch_height'] * config['patch_width'] * 3
        max_seq_len = (image_size // config['patch_height']) * (image_size // config['patch_width']) + 1
    elif patch_mode == 1:
        patch_vector_dim = image_size * config['patch_height'] * 3
        max_seq_len = (image_size // config['patch_height']) + 1
    elif patch_mode == 2:
        patch_vector_dim = image_size * config['patch_width'] * 3
        max_seq_len = (image_size // config['patch_width']) + 1
    config['max_seq_len'] = max_seq_len

    hidden_dim = config['hidden_dim']
    layers = config['layers']
    model_type = config['model_type']
    attn_heads = config['attn_heads']
    cell_number = config['cell_number']
    batch_size = config.get('batch_size', 1)

    if model_type == "6":
        if cell_number == "1":
            cell_type = LRUCell
        elif cell_number == "2":
            cell_type = IndyGRUCell
        elif cell_number == "3":
            cell_type = IndRNNCell3
        elif cell_number == "4":
            cell_type = RRUCell
        else:
            cell_type = nn.RNNCell
    else:
        cell_type = None

    model = select_model(model_type, patch_vector_dim, hidden_dim, layers, max_seq_len, cell_type, attn_heads).to(device)
    lr = 0.0006
    if model_type in ['4', '7', '10', '12']: #MLPs
        lr = 0.0006
    elif model_type in ['1']: #LSTMs
        lr = 0.004
    elif model_type in ['2', '6']: #RNNs
        lr = 0.0006
    elif model_type in ['3']: #MinGRU
        lr = 0.0006
    elif model_type in ['5', '8', '9', '11']: #Tranformers
        lr = 0.0006
    if device.type == 'cuda':
        from deepspeed.ops.adam import FusedAdam
        optimizer = FusedAdam(model.parameters(), lr=lr)
        print("GPU detected, using Fused Adam")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        print("CPU detected, using regular Adam")
    criterion = nn.HuberLoss()
    if cont == True:
        model.load_state_dict(torch.load("model.pth", map_location=device))
    model.train()

    dataset = PatchImageDataset(
        config['image_dir'], 
        image_size, 
        patch_mode, 
        patch_size=None,  # not used when separate dimensions are provided
        patch_height=config.get('patch_height') if patch_mode in [0,1] else None,
        patch_width=config.get('patch_width') if patch_mode in [0,2] else None
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    step = 0
    for epoch in range(config['epochs']):
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        for input_seq_tensor, target_seq_tensor in data_loader:
            input_seq_tensor = input_seq_tensor.to(device)
            target_seq_tensor = target_seq_tensor.to(device)

            optimizer.zero_grad()
            res = model(input_seq_tensor)
            output = res[0] if isinstance(res, tuple) else res
            loss = criterion(output, target_seq_tensor) * 100
            loss.backward()
            optimizer.step()

            lr_current = optimizer.param_groups[0]['lr']
            if isinstance(lr_current, torch.Tensor):
                lr_current = lr_current.cpu().item()
            print(f"Step {step}, Loss: {loss.item():.4f}, LR: {lr_current:.8f}")
            step += 1

            if step % 250 == 0:
                model.eval()
                sample_img = generate_image(model, config, device)
                sample_filename = f"LSTMimage/sample_step_{step}.png"
                save_sample_image(sample_img, sample_filename)
                torch.save(model.state_dict(), "model.pth")
                with open("config.json", "w") as f:
                    json.dump(config, f, indent=4)
                print(f"Checkpoint saved at step {step}.")
                model.train()

    torch.save(model.state_dict(), "model.pth")
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)
    print("Training complete. Model and configuration saved.")


def sample_model():
    if not os.path.exists("config.json") or not os.path.exists("model.pth"):
        print("Config or model checkpoint not found. Please train the model first.")
        return

    for filename in os.listdir("LSTMimage"):
        file_path = os.path.join("LSTMimage", filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)
    
    with open("config.json", "r") as f:
        config = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = config['image_size']
    patch_mode = config['patch_mode']
    if patch_mode == 0:
        patch_vector_dim = config['patch_height'] * config['patch_width'] * 3
    elif patch_mode == 1:
        patch_vector_dim = image_size * config['patch_height'] * 3
    elif patch_mode == 2:
        patch_vector_dim = image_size * config['patch_width'] * 3
    hidden_dim = config['hidden_dim']
    layers = config['layers']
    attn_heads = config['attn_heads']
    model_type = config['model_type']
    max_seq_len = config['max_seq_len']
    
    model = select_model(model_type, patch_vector_dim, hidden_dim, layers, max_seq_len, attn_heads=attn_heads).to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    num_samples = int(input("Enter number of samples to generate: "))
    samples = []
    for i in range(num_samples):
        with torch.no_grad():
            sample_img = generate_image(model, config, device)
        file_name = f"LSTMimage/sample_{i}.png"
        save_sample_image(sample_img, file_name)
        samples.append(sample_img)
    
    cols = int(math.ceil(math.sqrt(num_samples)))
    rows = int(math.ceil(num_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if num_samples > 1 else [axes]
    for idx, ax in enumerate(axes):
        if idx < len(samples):
            ax.imshow(samples[idx])
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()


import os
import shutil
from PIL import Image
from tqdm import tqdm

def resize_images(image_dir, image_size):
    resized_dir = os.path.join(os.path.dirname(image_dir), "resized")
    if os.path.exists(resized_dir):
        shutil.rmtree(resized_dir)
    os.makedirs(resized_dir)
    
    # Use tqdm to display progress bar over the list of files.
    for filename in tqdm(os.listdir(image_dir), desc="Resizing images"):
        img_path = os.path.join(image_dir, filename)
        try:
            img = Image.open(img_path)
            img = img.resize((image_size, image_size), Image.LANCZOS)
            img.save(os.path.join(resized_dir, filename))
        except Exception as e:
            print(f"Skipping {filename}: {e}")
    
    print(f"Resized images saved in: {resized_dir}")
    return resized_dir



def main():
    mode = input("Enter mode (train/continue/sample): ").strip().lower()
    if mode in ["train", "t", "0"]:
        imdir = input("Enter image directory: ").strip()
        print("Choose patch mode:")
        print("  (0) Classic patches (grid mode with user-specified height and width)")
        print("  (1) Rowwise patches (each patch covers a horizontal strip)")
        print("  (2) Columnwise patches (each patch covers a vertical strip)")
        patch_mode = input("Enter patch mode (0/1/2): ").strip()
        if patch_mode not in ["0", "1", "2"]:
            print("Invalid patch mode. Defaulting to classic (0).")
            patch_mode = "0"
        patch_mode = int(patch_mode)

        image_size = int(input("Enter image size (square): ").strip())
        if patch_mode == 0:
            patch_height = int(input("Enter patch height: ").strip())
            patch_width = int(input("Enter patch width: ").strip())
        elif patch_mode == 1:
            patch_height = int(input("Enter patch height for rowwise patches: ").strip())
            patch_width = None
        elif patch_mode == 2:
            patch_width = int(input("Enter patch width for columnwise patches: ").strip())
            patch_height = None

        model_type = input(
            "Choose model:\n(1) Long Short Term Memory\n(2) Gated Recurrent Unit\n(3) minGRU\n(4) Multi-Layered Perceptro\n(5) GPT-2 Transformer\n"
            "(6) RNN (will choose cell type after this)\n(7) Residual MLP\n"
            "(8) AliBi Transformer\n(9) Modern, Rotary pos embedded transformer\n(10) Graph Neural Network\n(11) Mamba\n(12) Pretty much Causal Convnet\n"
        ).strip()
        if model_type == "6":
            cell_number = input("Choose RNN cell\n(1) Light recurrent unit\n(2) IndyGRU\n(3) IndRNN\n(4) RRU\n ").strip()
        else:
            cell_number = "0"
        
        config = {
            'image_dir': imdir,
            'image_size': image_size,
            'patch_mode': patch_mode,
            'hidden_dim': int(input("Enter hidden dimension: ").strip()),
            'attn_heads': int(input("Enter number of attention heads: ").strip()) if model_type in ['5', '8', '9', '11'] else 8,
            'layers': int(input("Enter layer count: ").strip()),
            'epochs': int(input("Enter number of epochs: ").strip()),
            'model_type': model_type,
            'cell_number': cell_number,
            'batch_size': int(input("Enter batch size: ").strip()),
            'denoise': input("Denoise final generated image? (y/n): ").strip().lower() == 'y'
        }
        
        # Set patch dimensions based on mode.
        if patch_mode == 0:
            config['patch_height'] = patch_height
            config['patch_width'] = patch_width
        elif patch_mode == 1:
            config['patch_height'] = patch_height
        elif patch_mode == 2:
            config['patch_width'] = patch_width
        
        if input("Pre-resize images? (y/n): ").strip().lower() == 'y':
            config['image_dir'] = resize_images(imdir, image_size)
        
        train_model(config, False)

    elif mode in ["continue", "c", "1"]:
        # Continue training from previous config and checkpoint.
        if not os.path.exists("config.json") or not os.path.exists("model.pth"):
            print("Config or model checkpoint not found. Please train the model first.")
            return
        with open("config.json", "r") as f:
            config = json.load(f)
        new_imdir = input("Enter dataset (image) directory (continue training): ").strip()
        new_batch_size = int(input("Enter batch size (continue training): ").strip())
        config['image_dir'] = new_imdir
        config['batch_size'] = new_batch_size
        train_model(config, True)

    elif mode in ["sample", "s", "2"]:
        sample_model()
    else:
        print("Invalid mode. Please choose 'train', 'continue' or 'sample'.")


if __name__ == "__main__":
    main()
