# Multimodal Voice Phishing Detection System

## Project Overview
This project is a **multimodal framework for detecting voice phishing by jointly analyzing text and audio information**.  
It integrates a KoBERT-based text classifier and a CNN–BiLSTM-based synthetic voice detector, combined with speaker diarization (Resemblyzer + K-Means), Whisper STT, and weighted score fusion (Text:Voice = 8:2) to achieve reliable detection in real-world call environments.  

Traditional **keyword-based detection models** often misclassified conversations as normal when phishing-specific keywords were absent. To overcome this limitation, our system adopts a **context-aware text analysis** combined with **synthetic voice detection**, achieving more accurate and robust phishing detection.


## Key Features
- **Text Analysis**: Whisper STT → KoBERT classification  
- **Voice Analysis**: MFCC → CNN–BiLSTM synthetic voice detection  
- **Speaker Diarization**: Resemblyzer embeddings + K-Means clustering  
- **Multimodal Fusion**: 0.8 × Text + 0.2 × Voice final score  
- **Visualization**: Self-Attention–based token/pattern visualization  


## Project Structure (Simplified)
```text
ML/
├── KoBERTModel/         # Text classification module
├── deepvoice_detection/ # Synthetic voice detection module
├── speaker_analysis/    # Speaker diarization and STT module
├── figure/              # Visualization materials
├── static/              # Datasets and logs (csv)
├── templates/           # Web interface templates
├── test/                # Experimental and validation scripts
├── server.py            # Server execution script
├── shared_model_loader.py
├── requirements.txt
└── Dockerfile
```

## Performance Results

- **Validation (200 sentences)**: Accuracy, Precision, Recall, F1 = 100%  
- **Real-world calls (100 samples)**: 8:2 fusion model proved most stable  
- **Attention Analysis**:  
  - Normal → distributed attention patterns  
  - Phishing → concentrated on finance, command, and urgency keywords  


## Architecture (Screenshots & Descriptions)

### End-to-End Pipeline
<img width="4913" height="2268" alt="Architecture" src="https://github.com/user-attachments/assets/dea83e4a-d0a0-4b06-bc35-90d016b1db51" />  

The multimodal system processes incoming audio through **speaker diarization**, segmenting by speaker turns. Each utterance is then analyzed in parallel via **text analysis (KoBERT)** and **voice analysis (CNN–BiLSTM)**. The resulting scores are fused to produce the final phishing decision.


### Text Analysis Model (KoBERT-based)
<img width="4913" height="2268" alt="LLM_Model" src="https://github.com/user-attachments/assets/d2cc1636-518d-43aa-a32e-df14416dd633" />  

Text input is embedded with KoBERT and processed through a **Self-Attention–based Transformer Encoder**. Attention visualization highlights key tokens, and the output is passed through Fully Connected Layers with Softmax to generate phishing/normal probabilities.


### Voice Analysis Model (CNN–BiLSTM-based)
<img width="4913" height="2268" alt="1D_CNN" src="https://github.com/user-attachments/assets/12fbb60e-3c33-479e-a3b8-1ca9e00c2adc" />  

Audio is converted into **MFCC features**, processed by 1D CNN layers to extract local patterns, followed by BiLSTM layers to capture temporal dependencies. An attention layer emphasizes important segments, and a Fully Connected Layer with Sigmoid outputs the synthetic/real decision.


## Limitations
- Speaker diarization is sensitive to noise and audio quality  
- Static fusion ratio → requires context-adaptive dynamic fusion  


## Future Work
- Apply Whisper-based fine-grained multi-speaker separation  
- Research on dynamic multimodal weighting strategies  
- Expand to diverse phishing scenarios  


## References
- **Data**: Financial Supervisory Service (Korea), AI Hub  
- **Related Code**: [so-vits-svc-fork](https://github.com/voicepaw/so-vits-svc-fork)
- 

## Specifications
- **GPU**: RTX 3090 (24GB)  
- **System**: Windows 11  
- **PyTorch**: 1.10.1  
