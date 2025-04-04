# PTX Assistant - Chainlit App Setup Guide

This guide will help you set up and run the PTX Assistant Chainlit application, a document-based question-answering system.

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)
- Git
- CUDA-compatible GPU (recommended for better performance)
- Docker (optional, for containerized deployment)

## Installation Steps

1. **Clone the Repository**
   ```bash
   git clone [repository-url]
   cd ptx_assistant
   ```

2. **Create a Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   The main dependencies include:
   - chainlit
   - langchain and its components
   - transformers
   - sentence-transformers
   - pydantic
   - python-dotenv
   - and other necessary packages

## Configuration

   - `chat.config.yaml`: Contains chat configuration settings
   - `.config\config.toml`: Chainlit-specific configuration
   - `chainlit.md`: Chainlit-specific documentation

## Project Structure
```
ptx_assistant/
├── assistant/
│   ├── .chainlit/        # Chainlit configuration
│   ├── .files/           # Runtime Temporary files
│   ├── src/              # Source code
│   ├── models/           # Model downloads
│   ├── public/           # Public assets
│   ├── main.py           # Main application file
│   ├── config.py         # Configuration script (DO NOT MODIFY)
│   └── chat.config.yaml  # Chat configuration
├── requirements.txt      # Python dependencies
└── Dockerfile            # Container configuration
```

## Running the Application

### Local Development

1. **Start the Chainlit App**
   ```bash
   cd assistant
   chainlit run main.py
   ```

2. **Access the Application**
   - Open your web browser
   - Navigate to `http://localhost:8000`. Normally, it auto-opens in your default browser.

### Docker Deployment

1. **Build the Docker Image**
   ```bash
   docker build -t ptx-assistant .
   ```

2. **Run the Container**
   ```bash
   docker run -p 80:80 ptx-assistant
   ```

3. **Access the Application**
   - Open your web browser
   - Navigate to `http://localhost:80`

## Development

### Common Issues and Solutions

1. **CUDA Memory Issues**
   - If you encounter CUDA out of memory errors, try reducing the batch size
   - Use the following command to check GPU memory:
     ```bash
     nvidia-smi
     ```

2. **Dependency Conflicts**
   - If you encounter dependency conflicts, try creating a fresh virtual environment
   - Make sure to install packages in the correct order as specified in requirements.txt

3. **Model Loading Issues**
   - Ensure the model path in `.env` is correct
   - Check if the model files are complete and not corrupted

## Performance Optimization

1. **GPU Acceleration**
   - Ensure CUDA is properly installed
   - Use appropriate batch sizes for your GPU memory

2. **Memory Management**
   - Monitor memory usage during inference
   - Use appropriate model quantization if needed

## Support

For additional support or questions:
- Check the project documentation
- Open an issue on the repository
- Contact the development team
