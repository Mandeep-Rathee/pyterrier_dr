import pyterrier as pt
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import pyterrier_alpha as pta
from typing import List, Optional

from pyterrier_dr.biencoder import BiEncoder


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Pool the last token from the model output.
    Handles both left and right padding.
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format query with task instruction."""
    return f'Instruct: {task_description}\nQuery: {query}'


class Qwen(BiEncoder):
    """
    Qwen3-Embedding bi-encoder for PyTerrier.
    
    This class provides a PyTerrier interface to the Qwen3-Embedding model,
    following the same pattern as other dense retrieval models in pyterrier_dr.
    """
    
    def __init__(
        self, 
        model_name='Qwen/Qwen3-Embedding-0.6B', 
        batch_size=32, 
        max_length=8192, 
        text_field='text', 
        verbose=False, 
        device=None,
        use_fp16=False,
        task_description='Given a web search query, retrieve relevant passages that answer the query',
        add_instruction_to_query=True
    ):
        """
        Initialize the Qwen encoder.
        
        Args:
            model_name: The Hugging Face model name/path
            batch_size: Default batch size for encoding
            max_length: Maximum sequence length
            text_field: Field name in dataframes containing document text
            verbose: Whether to show progress bars
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            use_fp16: Whether to use FP16 precision
            task_description: Task description to prepend to queries
            add_instruction_to_query: Whether to add instruction to queries
        """
        super().__init__(batch_size=batch_size, text_field=text_field, verbose=verbose)
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self.task_description = task_description
        self.add_instruction_to_query = add_instruction_to_query
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Import transformers
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError("Qwen encoder requires transformers. Install with: pip install transformers")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        
        if use_fp16 and self.device.type == 'cuda':
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        
        self.model.eval()

    def __repr__(self):
        return f'Qwen({repr(self.model_name)})'
    
    def _encode_batch(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Internal method to encode a batch of texts.
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding (uses self.batch_size if None)
            
        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                batch_dict = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
                
                # Get model outputs
                outputs = self.model(**batch_dict)
                
                # Pool last token
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                
                # Normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                # Move to CPU and convert to numpy
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def encode_queries(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Encode query texts into dense vectors.
        
        Args:
            texts: List of query strings
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of query embeddings
        """
        # Add instruction to queries if enabled
        if self.add_instruction_to_query:
            texts = [get_detailed_instruct(self.task_description, text) for text in texts]
        
        return self._encode_batch(texts, batch_size)
    
    def encode_docs(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Encode document texts into dense vectors.
        
        Args:
            texts: List of document strings
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of document embeddings
        """
        # No instruction needed for documents
        return self._encode_batch(texts, batch_size)
    
    def query_encoder(self, verbose=None, batch_size=None):
        """Returns a transformer for encoding queries."""
        return QwenQueryEncoder(self, verbose=verbose, batch_size=batch_size)
    
    def doc_encoder(self, verbose=None, batch_size=None):
        """Returns a transformer for encoding documents."""
        return QwenDocEncoder(self, verbose=verbose, batch_size=batch_size)


class QwenQueryEncoder(pt.Transformer):
    """PyTerrier transformer for encoding queries using Qwen."""
    
    def __init__(self, qwen_factory: Qwen, verbose=None, batch_size=None, max_length=None):
        self.qwen_factory = qwen_factory
        self.verbose = verbose if verbose is not None else qwen_factory.verbose
        self.batch_size = batch_size if batch_size is not None else qwen_factory.batch_size
        self.max_length = max_length if max_length is not None else qwen_factory.max_length
    
    def encode(self, texts):
        return self.qwen_factory.encode_queries(list(texts), batch_size=self.batch_size)
    
    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pta.validate.columns(inp, includes=['query'])
        
        # Handle empty input
        if len(inp) == 0:
            return inp.assign(query_vec=[])
        
        # Get unique queries
        it = inp['query'].values
        it, inv = np.unique(it, return_inverse=True)
        
        if self.verbose:
            it = pt.tqdm(it, desc='Encoding Queries', unit='query')
        
        # Encode queries
        query_vecs = self.encode(it)
        
        # Assign back to dataframe using inverse mapping
        return inp.assign(query_vec=[query_vecs[i] for i in inv])
    
    def __repr__(self):
        return f'{repr(self.qwen_factory)}.query_encoder()'


class QwenDocEncoder(pt.Transformer):
    """PyTerrier transformer for encoding documents using Qwen."""
    
    def __init__(self, qwen_factory: Qwen, verbose=None, batch_size=None, max_length=None):
        self.qwen_factory = qwen_factory
        self.verbose = verbose if verbose is not None else qwen_factory.verbose
        self.batch_size = batch_size if batch_size is not None else qwen_factory.batch_size
        self.max_length = max_length if max_length is not None else qwen_factory.max_length
    
    def encode(self, texts):
        return self.qwen_factory.encode_docs(list(texts), batch_size=self.batch_size)
    
    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        # Validate input columns
        pta.validate.columns(inp, includes=[self.qwen_factory.text_field])
        
        # Handle empty input
        if len(inp) == 0:
            return inp.assign(doc_vec=[])
        
        # Get document texts
        it = inp[self.qwen_factory.text_field]
        
        if self.verbose:
            it = pt.tqdm(it, desc='Encoding Documents', unit='doc')
        
        # Encode documents
        doc_vecs = self.encode(it)
        
        # Assign back to dataframe
        return inp.assign(doc_vec=list(doc_vecs))
    
    def __repr__(self):
        return f'{repr(self.qwen_factory)}.doc_encoder()'

