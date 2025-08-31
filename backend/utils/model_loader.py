import torch
from transformers import BertTokenizer
from models import EnhancedBERTModel, ELMOBert, ELMoTransformerModel, FiveEmbeddingModel
from utils import DriveModelLoader


def setup_device():
    """Setup and return the best available device"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("📱 Using MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    return device


def load_tokenizer():
    """Load and return BERT tokenizer"""
    print("🔤 Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("✅ Tokenizer loaded")
    return tokenizer


def load_enhanced_bert_model(device):
    """Load Enhanced BERT model from Google Drive"""
    print("🏗️ Initializing Enhanced BERT model architecture...")
    model = EnhancedBERTModel(num_classes=4, dropout_rate=0.3)
    
    print("📦 Loading Enhanced BERT model weights from Google Drive...")
    try:
        drive_loader = DriveModelLoader()
        model_path = drive_loader.download_enhanced_bert_model()
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Enhanced BERT model weights loaded successfully from {model_path}")
        
    except Exception as e:
        print(f"❌ Error loading Enhanced BERT model: {str(e)}")
        raise
    
    model = model.to(device)
    model.eval()
    print(f"🎯 Enhanced BERT model ready on {device}")
    
    return model


def load_elmo_bert_model(device):
    """Load ELMo+BERT model from Google Drive"""
    print("🏗️ Initializing ELMo+BERT model architecture...")
    model = ELMOBert(num_classes=4)
    
    print("📦 Loading ELMo+BERT model weights from Google Drive...")
    try:
        drive_loader = DriveModelLoader()
        model_path = drive_loader.download_elmo_bert_model()
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ ELMo+BERT model weights loaded successfully from {model_path}")
        
    except Exception as e:
        print(f"❌ Error loading ELMo+BERT model: {str(e)}")
        raise
    
    model = model.to(device)
    model.eval()
    print(f"🎯 ELMo+BERT model ready on {device}")
    
    return model


def load_elmo_transformer_model(device):
    """Load ELMo Transformer model from Google Drive"""
    print("🏗️ Initializing ELMo Transformer model architecture...")
    model = ELMoTransformerModel(num_classes=4)
    
    print("📦 Loading ELMo Transformer model weights from Google Drive...")
    try:
        drive_loader = DriveModelLoader()
        model_path = drive_loader.download_elmo_transformer_model()
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ ELMo Transformer model weights loaded successfully from {model_path}")
        
    except Exception as e:
        print(f"❌ Error loading ELMo Transformer model: {str(e)}")
        raise
    
    model = model.to(device)
    model.eval()
    print(f"🎯 ELMo Transformer model ready on {device}")
    
    return model


def load_5_embedding_model(device):
    """Load 5-Embedding model from Google Drive"""
    print("🏗️ Initializing 5-Embedding model architecture...")
    model = FiveEmbeddingModel(num_classes=4)
    
    print("📦 Loading 5-Embedding model weights from Google Drive...")
    try:
        drive_loader = DriveModelLoader()
        model_path = drive_loader.download_5_embedding_model()
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 5-Embedding model weights loaded successfully from {model_path}")
        
    except Exception as e:
        print(f"❌ Error loading 5-Embedding model: {str(e)}")
        raise
    
    model = model.to(device)
    model.eval()
    print(f"🎯 5-Embedding model ready on {device}")
    
    return model
