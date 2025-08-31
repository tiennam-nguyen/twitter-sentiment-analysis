import torch
import torch.nn.functional as F
from .preprocessing import intelligent_preprocess  # ✅ Use the actual function
from transformers import AutoTokenizer
import logging
import stanza

# Global variables for expensive initialization
_stanza_nlp = None
_pos_vocab = None

def get_stanza_nlp():
    """Get or initialize Stanza NLP pipeline."""
    global _stanza_nlp
    if _stanza_nlp is None:
        try:
            import stanza
            # Try GPU first, fall back to CPU
            try:
                _stanza_nlp = stanza.Pipeline('en', use_gpu=True, download_method=None)
                logging.info("Stanza initialized with GPU")
            except:
                _stanza_nlp = stanza.Pipeline('en', use_gpu=False, download_method=None)
                logging.info("Stanza initialized with CPU")
        except Exception as e:
            logging.error(f"Failed to initialize Stanza: {e}")
            raise e
    return _stanza_nlp

def get_pos_vocab():
    """Get or initialize POS vocabulary."""
    global _pos_vocab
    if _pos_vocab is None:
        _pos_vocab = {
            'NOUN': 1, 'VERB': 2, 'ADJ': 3, 'ADV': 4, 'PRON': 5,
            'DET': 6, 'ADP': 7, 'NUM': 8, 'CONJ': 9, 'PRT': 10,
            'X': 11, 'PUNCT': 12, 'INTJ': 13, 'SYM': 14, 'PROPN': 15,
            'CCONJ': 16, 'SCONJ': 17, 'AUX': 18
        }
    return _pos_vocab


def predict_sentiment_enhanced_bert(model, text, tokenizer, device, max_length=128):
    """Internal prediction function"""
    model.eval()

    processed_text = intelligent_preprocess(text)  # ✅ Fixed

    encoding = tokenizer(
        processed_text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask, token_type_ids)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "probabilities": {
            "class_0": float(probabilities[0][0]),
            "class_1": float(probabilities[0][1]),
            "class_2": float(probabilities[0][2]),
            "class_3": float(probabilities[0][3])
        },
        "input_text": text,
        "processed_text": processed_text
    }


def predict_sentiment_elmo_bert(model, text, tokenizer, device, max_length=64):
    """
    Prediction function for ELMo+BERT model
    
    Args:
        model: ELMo+BERT model instance
        text: Input text to analyze
        tokenizer: BERT tokenizer
        device: Device to run inference on
        max_length: Maximum sequence length (default: 64 for ELMo+BERT)
        
    Returns:
        Dictionary containing prediction results
    """
    model.eval()

    processed_text = intelligent_preprocess(text)  # ✅ Fixed

    # Prepare input in the format expected by ELMo+BERT model
    encoding = tokenizer(
        [processed_text],  # Note: needs to be a list for batch processing
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Create batch dictionary as expected by DICETModel
    batch = {
        'sentences': [processed_text],  # ELMo needs the raw sentences
        'input_ids': encoding['input_ids'].to(device),
        'attention_mask': encoding['attention_mask'].to(device),
        'token_type_ids': encoding['token_type_ids'].to(device)
    }
    
    with torch.no_grad():
        outputs = model(batch)
        logits = outputs["logits"]
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "probabilities": {
            "class_0": float(probabilities[0][0]),
            "class_1": float(probabilities[0][1]),
            "class_2": float(probabilities[0][2]),
            "class_3": float(probabilities[0][3])
        },
        "input_text": text,
        "processed_text": processed_text,
    }


def predict_sentiment_elmo_transformer(model, text, device):
    """
    Prediction function for ELMo Transformer model
    
    Args:
        model: ELMo Transformer model instance
        text: Input text to analyze
        device: Device to run inference on
        
    Returns:
        Dictionary containing prediction results
    """
    model.eval()

    processed_text = intelligent_preprocess(text)  # ✅ Fixed

    with torch.no_grad():
        # ELMo Transformer model expects a list of texts
        outputs = model([processed_text])
        logits = outputs["logits"]
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "probabilities": {
            "class_0": float(probabilities[0][0]),
            "class_1": float(probabilities[0][1]),
            "class_2": float(probabilities[0][2]),
            "class_3": float(probabilities[0][3])
        },
        "input_text": text,
        "processed_text": processed_text,
        "attention_weights": outputs.get("attn", None)
    }


def predict_sentiment_5_embedding(model, text, tokenizer, device, max_length=64):
    """
    Complete prediction function for 5-Embedding model
    Requires full preprocessing pipeline for all 6 embedding types
    
    Args:
        model: 5-Embedding model instance (DICETModel)
        text: Input text to analyze
        tokenizer: BERT tokenizer
        device: Device to run inference on
        max_length: Maximum sequence length for BERT
        
    Returns:
        Dictionary containing prediction results
    """
    model.eval()
    
    # Initialize required components (this should be done once at startup, not per prediction)
    try:
        # POS vocabulary from the notebook
        pos_vocab = {
            '<pad>': 0, 'NOUN': 1, 'VERB': 2, 'ADJ': 3, 'ADV': 4, 'PROPN': 5, 
            'PUNCT': 6, 'DET': 7, 'ADP': 8, 'CCONJ': 9, 'SCONJ': 10, 'PRON': 11, 
            'AUX': 12, 'NUM': 13, 'PART': 14, 'SYM': 15, 'X': 16, 'INTJ': 17
        }
        
        # Initialize Stanza (this is expensive - should be cached)
        try:
            nlp = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=torch.cuda.is_available(), verbose=False)
        except Exception:
            nlp = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False, verbose=False)
            
    except Exception as e:
        return {
            "error": f"Failed to initialize preprocessing components: {str(e)}",
            "input_text": text,
            "processed_text": intelligent_preprocess(text)
        }

    processed_text = intelligent_preprocess(text)
    
    try:
        # 1. BERT preprocessing
        bert_encoding = tokenizer(
            [processed_text],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # 2. GloVe preprocessing (using same tokenizer but different max_length)
        glove_encoding = tokenizer(
            [processed_text],
            truncation=True,
            padding='max_length',
            max_length=70,  # From notebook
            return_tensors='pt'
        )
        
        # 3. POS preprocessing
        doc = nlp(processed_text)
        pos_tokens = [word.upos for sent in doc.sentences for word in sent.words]
        pos_ids_list = [pos_vocab.get(tag, 0) for tag in pos_tokens]
        
        # Pad POS sequence to match BERT length
        if len(pos_ids_list) > max_length:
            pos_ids_list = pos_ids_list[:max_length]
        else:
            pos_ids_list.extend([0] * (max_length - len(pos_ids_list)))
            
        pos_ids = torch.tensor([pos_ids_list], dtype=torch.long)
        
        # 4. Character preprocessing
        words = processed_text.split()
        if not words:
            words = ['']  # Handle empty text
            
        max_word_len = max([len(w) for w in words] + [1])  # At least length 1
        word_char_ids = []
        
        for word in words:
            char_indices = [ord(c) for c in word if ord(c) < 128]  # ASCII only
            char_indices = char_indices[:max_word_len]  # Truncate if too long
            padding = [0] * (max_word_len - len(char_indices))
            word_char_ids.append(char_indices + padding)
        
        # Pad or truncate to match BERT sequence length
        if len(word_char_ids) > max_length:
            word_char_ids = word_char_ids[:max_length]
        else:
            empty_word = [0] * max_word_len
            word_char_ids.extend([empty_word] * (max_length - len(word_char_ids)))
            
        char_ids = torch.tensor([word_char_ids], dtype=torch.long)
        
        # 5. Create complete batch
        batch = {
            'sentences': [processed_text],  # For ELMo
            'input_ids': bert_encoding['input_ids'].to(device),
            'attention_mask': bert_encoding['attention_mask'].to(device),
            'token_type_ids': bert_encoding['token_type_ids'].to(device),
            'token_ids': glove_encoding['input_ids'].to(device),  # For GloVe
            'lex_ids': glove_encoding['input_ids'].clone().to(device),  # For Lexicon (same as GloVe)
            'pos_ids': pos_ids.to(device),
            'char_ids': char_ids.to(device),
        }
        
        # 6. Run inference
        with torch.no_grad():
            outputs = model(batch)
            logits = outputs["logits"]
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            return {
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "probabilities": {
                    "class_0": float(probabilities[0][0]),
                    "class_1": float(probabilities[0][1]),
                    "class_2": float(probabilities[0][2]),
                    "class_3": float(probabilities[0][3])
                },
                "input_text": text,
                "processed_text": processed_text,
                "attention_weights": outputs.get("attn", None)
            }
            
    except Exception as e:
        return {
            "error": f"5-Embedding model prediction failed: {str(e)}",
            "input_text": text,
            "processed_text": processed_text
        }


def initialize_5_embedding_dependencies():
    """
    Initialize expensive dependencies once at startup
    Returns a dictionary of initialized components
    """
    try:
        pos_vocab = {
            '<pad>': 0, 'NOUN': 1, 'VERB': 2, 'ADJ': 3, 'ADV': 4, 'PROPN': 5, 
            'PUNCT': 6, 'DET': 7, 'ADP': 8, 'CCONJ': 9, 'SCONJ': 10, 'PRON': 11, 
            'AUX': 12, 'NUM': 13, 'PART': 14, 'SYM': 15, 'X': 16, 'INTJ': 17
        }
        
        try:
            nlp = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=torch.cuda.is_available(), verbose=False)
        except Exception:
            nlp = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False, verbose=False)
            
        return {
            'pos_vocab': pos_vocab,
            'nlp': nlp,
            'initialized': True
        }
    except Exception as e:
        print(f"Warning: Failed to initialize 5-embedding dependencies: {e}")
        return {'initialized': False, 'error': str(e)}


def predict_sentiment_5_embedding_optimized(model, text, tokenizer, device, dependencies, max_length=64):
    """
    Optimized version that uses pre-initialized dependencies
    
    Args:
        model: 5-Embedding model instance
        text: Input text to analyze
        tokenizer: BERT tokenizer
        device: Device to run inference on
        dependencies: Pre-initialized dependencies from initialize_5_embedding_dependencies()
        max_length: Maximum sequence length
        
    Returns:
        Dictionary containing prediction results
    """
    if not dependencies.get('initialized', False):
        return {
            "error": "5-Embedding dependencies not initialized properly",
            "input_text": text,
            "processed_text": intelligent_preprocess(text)
        }
    
    model.eval()
    processed_text = intelligent_preprocess(text)
    
    try:
        pos_vocab = dependencies['pos_vocab']
        nlp = dependencies['nlp']
        
        # Same preprocessing logic as above but using pre-initialized components
        bert_encoding = tokenizer([processed_text], truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
        glove_encoding = tokenizer([processed_text], truncation=True, padding='max_length', max_length=70, return_tensors='pt')
        
        # POS processing
        doc = nlp(processed_text)
        pos_tokens = [word.upos for sent in doc.sentences for word in sent.words]
        pos_ids_list = [pos_vocab.get(tag, 0) for tag in pos_tokens]
        
        if len(pos_ids_list) > max_length:
            pos_ids_list = pos_ids_list[:max_length]
        else:
            pos_ids_list.extend([0] * (max_length - len(pos_ids_list)))
        pos_ids = torch.tensor([pos_ids_list], dtype=torch.long)
        
        # Character processing
        words = processed_text.split() or ['']
        max_word_len = max([len(w) for w in words] + [1])
        word_char_ids = []
        
        for word in words:
            char_indices = [ord(c) for c in word if ord(c) < 128][:max_word_len]
            padding = [0] * (max_word_len - len(char_indices))
            word_char_ids.append(char_indices + padding)
        
        if len(word_char_ids) > max_length:
            word_char_ids = word_char_ids[:max_length]
        else:
            empty_word = [0] * max_word_len
            word_char_ids.extend([empty_word] * (max_length - len(word_char_ids)))
        char_ids = torch.tensor([word_char_ids], dtype=torch.long)
        
        # Create batch and run inference
        batch = {
            'sentences': [processed_text],
            'input_ids': bert_encoding['input_ids'].to(device),
            'attention_mask': bert_encoding['attention_mask'].to(device),
            'token_type_ids': bert_encoding['token_type_ids'].to(device),
            'token_ids': glove_encoding['input_ids'].to(device),
            'lex_ids': glove_encoding['input_ids'].clone().to(device),
            'pos_ids': pos_ids.to(device),
            'char_ids': char_ids.to(device),
        }
        
        with torch.no_grad():
            outputs = model(batch)
            logits = outputs["logits"]
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            return {
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "probabilities": {
                    "class_0": float(probabilities[0][0]),
                    "class_1": float(probabilities[0][1]),
                    "class_2": float(probabilities[0][2]),
                    "class_3": float(probabilities[0][3])
                },
                "input_text": text,
                "processed_text": processed_text,
                "attention_weights": outputs.get("attn", None)
            }
            
    except Exception as e:
        return {
            "error": f"5-Embedding model prediction failed: {str(e)}",
            "input_text": text,
            "processed_text": processed_text
        }