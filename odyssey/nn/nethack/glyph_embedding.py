from odyssey.nethack.describe import describe_glyph

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

import nle.nethack as nh

import os
import hashlib

BORDER_GLYPH = nh.MAX_GLYPH + 1

def get_max_glyph(include_border_glyph=False):
    return BORDER_GLYPH if include_border_glyph else nh.MAX_GLYPH

def create_glyph_embedding(
    embedding_dim,
    add_border_glyph=True,
    **embedding_kwargs
):
    return nn.Embedding(
        num_embeddings=get_max_glyph(include_border_glyph=add_border_glyph),
        embedding_dim=embedding_dim,
        **embedding_kwargs
    )

def create_glyph_text_embedding(
    embedding_dim,
    embedding_net: SentenceTransformer,
    describe_glyph_fn=describe_glyph,
    cache_folder: str = None,
    add_border_glyph=True,
    border_word="border",
    **embedding_kwargs
): 
    """
    Creates a glyph embedding using a pre-trained text embedding model.

    This function generates text descriptions for glyphs and uses a pre-trained SentenceTransformer to create embeddings for these descriptions. 
    A trainable linear layer then maps these embeddings to the final glyph embedding.

    Args:
        embedding_dim (int): The dimension of the embedding vector.
        embedding_net (SentenceTransformer): A pre-trained text embedding model.
        describe_glyph_fn (function, optional): Function to describe glyphs.
        cache_folder (str, optional): Path to a folder to store the pre-computed text embeddings for quick loading.
        add_border_glyph (bool, optional): Whether to include a border glyph in the embeddings.
        border_word (str, optional): The word to use for the border glyph description.
        **embedding_kwargs: Additional keyword arguments for the embedding layer.

    Returns:
        nn.Module: A module that maps glyph indices to embedding vectors.
    """

    embedding = None
    if cache_folder:
        # Hash the input parameters to create a unique identifier for the cache file
        hash_input = f"{embedding_dim}_{embedding_net.__class__.__name__}_{str(embedding_kwargs)}"
        hash = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

        cache_file_path = os.path.join(cache_folder, f"glyph_text_embeddings_{hash}.torch")
        if os.path.exists(cache_file_path):
            print(f"Loading cached glyph embeddings '{cache_file_path}'...")
            embedding_weights = torch.load(cache_file_path, weights_only=True)
            embedding = nn.Embedding(get_max_glyph(include_border_glyph=add_border_glyph), embedding_net.get_sentence_embedding_dimension())
            embedding.load_state_dict(embedding_weights)
    
    if embedding is None:
        print("Generating glyph descriptions...")
        glyph_descriptions = [
            describe_glyph_fn(glyph)
            for glyph in range(nh.MAX_GLYPH)
        ]
        if add_border_glyph:
            glyph_descriptions.append(border_word)

        print("Generating glyph sentence embeddings...")
        embedding = embedding_net.encode(glyph_descriptions, show_progress_bar=True)
        embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding))

    if cache_folder and not os.path.exists(cache_file_path):
        os.makedirs(cache_folder, exist_ok=True)
        torch.save(embedding.state_dict(), cache_file_path)

    return nn.Sequential(
        embedding,
        nn.Linear(embedding.embedding_dim, embedding_dim)
    )

