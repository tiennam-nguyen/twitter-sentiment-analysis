# Twitter Sentiment Analysis

A project from HCMUT's ML IoT Lab comparing multiple deep learning architectures for sentiment analysis on noisy Twitter data. This system classifies tweets into 4 categories: Positive, Negative, Neutral, and Irrelevant.

This project was developed by the team "5 anh em siêu nhân" and advised by Trần Minh Huy.

## 🚀 Features

* **Multi-Model Comparison**: Experiments with four distinct architectures: BERT only, ELMo + BERT, Five-Embedding, and ELMo + Transformer (DialoGPT-based).
* **4-Class Classification**: Detects Positive, Negative, Neutral, and Irrelevant sentiments.
* **Intelligent Preprocessing**: A robust pipeline designed to handle noisy social media text.
* **FastAPI Backend**: High-performance REST API with CORS support.
* **React Frontend**: Modern, responsive user interface with real-time predictions.
* **Google Drive Integration**: Automatic model download from cloud storage.
* **Docker Support**: Fully containerized deployment.
* **Real-time Predictions**: Fast sentiment analysis with confidence scores.

## 📁 Project Structure

```bash
twitter_sentiment_bert_only/
├── backend/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── enhanced_bert.py         # Model 1: BERT only (with multi-head attention)
│   │   ├── elmo_bert.py             # Model 2: ELMo+BERT combined model architecture 
│   │   └── elmo_bert.ipynb          # Jupyter notebook with model development
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── drive_loader.py          # Google Drive API integration
│   │   ├── model_loader.py          # Model loading and initialization
│   │   ├── prediction.py            # Prediction pipeline
│   │   └── preprocessing.py         # Text preprocessing utilities
│   ├── server.py                    # FastAPI application
│   ├── requirements.txt             # Python dependencies
│   ├── Dockerfile                   # Docker configuration
│   ├── .env.example                 # Environment variables template
│   └── .gitignore                   # Git ignore rules
├── frontend/
│   ├── src/
│   │   ├── App.tsx                  # Main React sentiment analyzer UI
│   │   ├── components/ui/           # Reusable UI components
│   │   └── ...
│   ├── package.json                 # Node.js dependencies
│   ├── tailwind.config.ts           # Tailwind CSS configuration
│   └── ...
└── README.md
````

## 🛠️ Setup Instructions

### Prerequisites

  * Python 3.12+
  * Node.js 18+
  * Docker (optional)
  * Google Drive API credentials

### Backend Setup

1.  **Clone the repository**

    ```bash
    git clone [https://github.com/VoMinhKhoii/hcmut-project-cuoi-khoa.git](https://github.com/VoMinhKhoii/hcmut-project-cuoi-khoa.git)
    cd hcmut-project-cuoi-khoa/backend
    ```

2.  **Create virtual environment**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Setup Google Drive API**
    Get Google Drive API credentials from Google Cloud Console.
    Create a `.env` file in the `backend` directory:

    ```ini
    GOOGLE_DRIVE_FOLDER_ID=your_folder_id_here
    REFRESH_TOKEN=your_refresh_token_here
    CLIENT_ID=your_client_id_here
    CLIENT_SECRET=your_client_secret_here
    ```

5.  **Run the server**

    ```bash
    python server.py
    # Or with uvicorn
    uvicorn server:app --host 0.0.0.0 --port 8888
    ```

### Frontend Setup

1.  **Navigate to frontend directory**

    ```bash
    cd ../frontend
    ```

2.  **Install dependencies**

    ```bash
    npm install
    ```

3.  **Start development server**

    ```bash
    npm run dev
    ```

4.  **Access the application**

      * **Frontend**: `http://localhost:8080`
      * **Backend API**: `http://localhost:8888`
      * **API Documentation**: `http://localhost:8888/docs`

### Docker Setup (Recommended for Production)

1.  **Build and run with Docker**

    ```bash
    cd backend
    docker build -t sentiment-api .
    docker run -p 8888:8888 sentiment-api
    ```

2.  **Run frontend separately**

    ```bash
    cd frontend
    npm run dev
    ```

## 🔧 API Usage

> **Note:** The API provides endpoints for the "BERT only" and "ELMo + BERT" models, which were part of the broader comparison.

### Predict Sentiment (BERT only)

**POST** `/predict`

```json
{
  "text": "I love this product! It works amazing!"
}
```

**Response:**

```json
{
  "text": "I love this product! It works amazing!",
  "sentiment": "positive",
  "confidence": 0.9245,
  "probabilities": {
    "positive": 0.9245,
    "negative": 0.0123,
    "neutral": 0.0532,
    "irrelevant": 0.0100
  },
  "processed_text": "i love this product it works amazing"
}
```

### Predict Sentiment (ELMo+BERT)

**POST** `/predict-elmo-bert`

Uses the combined ELMo+BERT architecture, which was the best-performing model in the experiment.

```json
{
  "text": "This movie is absolutely fantastic!"
}
```

**Response:** Same format as above, but uses the ELMo+BERT model for prediction.

### Health Check

**GET** `/`

Returns API status and model information.

## 🧠 Model Architectures & Comparison

The core of this project was to find the best model for handling unstructured and noisy Twitter data, which includes slang, typos, emojis, and hashtags. We experimented with four architectures:

  * **BERT only**: A standard BERT (Bidirectional Encoder Representations from Transformers) model with a custom classification head, including Multi-Head Self-Attention and Residual Connections.
  * **ELMo + BERT**: A combined architecture that concatenates contextual embeddings from both ELMo and BERT. This combined embedding is then passed through a BiLSTM with Attention layer for final classification.
  * **ELMo + Transformer (DialoGPT-based)**: This model replaces the BERT embedding with one from DialoGPT (a conversational AI from GPT-2). The embedding is then passed to a BiLSTM with Attention layer.
  * **Five-Embedding**: The most complex model, which concatenates five different types of embeddings: GloVe, Part-of-Speech (POS), Lexicon, ELMo, and BERT. This rich, 2198-dimension embedding is also processed by a BiLSTM with Attention layer.

## 📊 Performance

The models were trained and evaluated, focusing on validation accuracy and loss. The dataset was noted to have an uneven distribution of sentiments (imbalanced data).

  * **Best Performance**: The **ELMo + BERT** model achieved the highest validation accuracy (approx. 98%) and the lowest validation loss (approx. 0.07).
  * **Runner-up**: The **Five-Embedding** model also performed very well, with the second-highest validation accuracy (approx. 96%).
  * **Conclusion**: Combining different powerful embeddings (like ELMo and BERT) proved more effective than using a single model (BERT only) or a more complex combination (Five-Embedding) for this specific task.

| Model | Embedding Dimension | Validation Accuracy (Approx.) | Validation Loss (Approx.) |
| :--- | :--- | :--- | :--- |
| **ELMo + BERT** | 1792 dims | \~98% | \~0.07 |
| **Five-Embedding** | 2198 dims | \~96% | \~0.12 |
| **ELMo + Transformer** | 1024 dims | \~95% | \~0.125 |
| **BERT only** | 768 dims | \~92% | \~0.28 |

## 🔄 Text Preprocessing Pipeline

To handle the noise of Twitter data, a multi-step intelligent preprocessing pipeline was used:

  * Spell Correction
  * Hashtag Processing
  * Sentiment Aware Tokenization
  * URL Handling (replaces URLs)
  * Mention Handling (processes @user)
  * Special Character Deletion

(Also includes emoji conversion, punctuation normalization, contraction expansion, and case normalization)

## 🌐 Frontend Features

  * **Real-time Analysis**: Instant sentiment prediction.
  * **Interactive UI**: Clean, modern interface built with React and Tailwind CSS.
  * **Confidence Visualization**: Progress bars showing prediction confidence.
  * **Example Texts**: Pre-loaded examples for quick testing.
  * **Error Handling**: Graceful error messages and loading states.
  * **Responsive Design**: Works on desktop and mobile devices.

## 📝 Environment Variables

Create a `.env` file in the `backend` directory:

```ini
# Google Drive Configuration
GOOGLE_DRIVE_FOLDER_ID=your_folder_id_containing_model
REFRESH_TOKEN=your_google_refresh_token
CLIENT_ID=your_google_client_id
CLIENT_SECRET=your_google_client_secret
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build the image
docker build -t sentiment-api ./backend

# Run the container
docker run -p 8888:8888 --env-file ./backend/.env sentiment-api
```

### Cloud Deployment

The application is ready for deployment on:

  * Google Cloud Run
  * AWS ECS
  * Azure Container Instances
  * Heroku

## 🤝 Contributing

1.  Fork the repository
2.  Create a feature branch (`git checkout -b feature/amazing-feature`)
3.  Commit your changes (`git commit -m 'Add some amazing feature'`)
4.  Push to the branch (`git push origin feature/amazing-feature`)
5.  Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 👥 Team Members ("5 anh em siêu nhân")

  * Vũ Trí Khải
  * Võ Minh Khôi - [VoMinhKhoii](https://github.com/VoMinhKhoii)
  * Nguyễn Tiến Nam - [tiennam-nguyen](https://github.com/tiennam-nguyen)
  * Hoàng Minh Hải
  * Đoàn Trần Quốc Việt

### Advisor / Mentor

  * Trần Minh Huy

## 🙏 Acknowledgments

  * HCMUT EE MACHINE LEARNING & IOT LAB (BK-tel ML IoT)
  * FastAPI and React communities for excellent frameworks
  * Google Drive API for model storage solution

## 📧 Contact

For questions or support, please contact:
GitHub: [@VoMinhKhoii](https://github.com/VoMinhKhoii)

-----

</p><p align="center">Ho Chi Minh City – 2025</p>
