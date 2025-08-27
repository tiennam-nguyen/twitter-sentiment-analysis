import torch
from transformers import BertTokenizer
from models import EnhancedBERTModel
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


def load_model(device, folder_id="1FbfKK1eEw9gS58KMHk4NCprfUIg3SpJM"):
    """Load model from Google Drive"""
    print("🏗️ Initializing model architecture...")
    model = EnhancedBERTModel(num_classes=4, dropout_rate=0.3)
    
    print("📦 Loading model weights from Google Drive...")
    try:
        drive_loader = DriveModelLoader()
        model_path = drive_loader.download_model_if_needed(folder_id)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Model weights loaded successfully from {model_path}")
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise
    
    model = model.to(device)
    model.eval()
    print(f"🎯 Model ready on {device}")
    
    return model


def initialize_sentiment_model():
    """Initialize complete sentiment analysis setup"""
    print("🚀 Initializing sentiment analysis model...")
    print("=" * 60)
    
    # Setup device
    device = setup_device()
    
    # Load tokenizer
    tokenizer = load_tokenizer()
    
    # Load model
    model = load_model(device)
    
    print("=" * 60)
    print("✅ Model initialization complete!")
    
    return model, tokenizer, device
