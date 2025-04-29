import torch
import torch.nn as nn

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def apply_model_to_sequence(model: nn.Module, sequences: torch.Tensor):
    """
    Applies a model to each element in a sequence.
    
    Args:
        model (nn.Module): The model to apply.
        sequence (torch.Tensor): The sequences of inputs of shape (batch_size, seq_len, *input_dim).
    Returns:
        torch.Tensor: The output of the model applied to each element in the sequence shape (batch_size, seq_len, *output_dim).
    """
    batch_size, seq_len, *input_dim = sequences.shape
    reshaped_sequences = sequences.view(batch_size * seq_len, *input_dim)
    reshaped_outputs = model(reshaped_sequences)
    output_dim = reshaped_outputs.shape[1:]
    outputs = reshaped_outputs.view(batch_size, seq_len, *output_dim)
    return outputs