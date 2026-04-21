# Cattle Breed Classification— AI Breed Identification

An AI-powered web application for identifying cattle and buffalo breeds from images using machine learning. This application provides a REST API backend and a modern web interface for breed classification.

## Features

- **AI Breed Classification**: Identifies 10 different cattle and buffalo breeds using ONNX model
- **Web Interface**: Modern, responsive frontend for easy image upload and results
- **REST API**: Flask-based API for programmatic access
- **Breed Information**: Detailed metadata for each breed including origin, purpose, and milk yield
- **Model Caching**: Automatic download and caching of AI models from Hugging Face
- **Cross-Origin Support**: CORS enabled for frontend integration

## Supported Breeds

### Cattle Breeds
- **Gir** - Indigenous dairy breed from Gujarat
- **Sahiwal** - High-yielding dairy breed from Punjab/Rajasthan
- **Ongole** - Dual-purpose breed from Andhra Pradesh
- **Kankrej** - Dual-purpose breed from Gujarat/Rajasthan
- **Tharparkar** - Dual-purpose breed from Rajasthan

### Buffalo Breeds
- **Murrah** - Premium dairy breed from Haryana/Punjab
- **Surti** - High-fat milk producer from Gujarat
- **Jaffarbadi** - Large-framed dairy breed from Gujarat
- **Mehsana** - Crossbred dairy breed from Gujarat
- **Bhadawari** - Indigenous breed from UP/MP

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone or download the project files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

### Web Interface
1. Open the application in your browser
2. Click "Upload Image" to select an animal photo
3. Wait for AI analysis (typically 2-3 seconds)
4. View breed identification results with confidence scores
5. Access detailed breed information and characteristics

### API Usage

The application provides a REST API for programmatic access:

#### POST /api/identify
Upload an image for breed identification.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `image` field with image file

**Response:**
```json
{
  "success": true,
  "breed": "Sahiwal",
  "confidence": 0.92,
  "metadata": {
    "type": "Cattle",
    "subtype": "Indigenous",
    "origin": "Punjab/Rajasthan",
    "purpose": "Dairy",
    "milk_yield": "8–14 L/day",
    "fat_pct": "4.8%"
  }
}
```

#### GET /api/breeds
Get information about all supported breeds.

#### GET /api/health
Check API health and model status.

## Project Structure

```
├── app.py                 # Flask application and API endpoints
├── requirements.txt       # Python dependencies
├── static/
│   └── index.html        # Web interface
├── model_cache/          # Cached AI models (auto-downloaded)
│   ├── model.onnx       # ONNX model file
│   └── prototypes.json  # Model metadata
└── uploads/             # Temporary uploaded images
```

## Technologies Used

- **Backend**: Flask, Python
- **AI/ML**: ONNX Runtime, PyTorch, TorchVision
- **Model**: vishnuamar/cattle-breed-classifier (Hugging Face)
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Image Processing**: Pillow (PIL)
- **API**: RESTful JSON API with CORS support

## Model Details

- **Architecture**: ONNX-compatible neural network
- **Input**: RGB images (224x224 pixels)
- **Output**: Breed classification with confidence scores
- **Training Data**: Diverse cattle and buffalo images
- **Accuracy**: High accuracy on standard test sets

## Development

### Running in Demo Mode
If ONNX Runtime is not available, the application runs in demo mode with mock predictions.

### Model Download
Models are automatically downloaded from Hugging Face on first run. Manual download is not required.

### File Upload Limits
- Maximum file size: 10 MB
- Supported formats: JPEG, PNG, WebP

## Contributing

This project is developed for agricultural and livestock management applications. Contributions are welcome for:

- Additional breed support
- Improved model accuracy
- UI/UX enhancements
- API documentation
- Testing and validation

## License

This project is open source and available under the MIT License.

## Contact

For questions or support, please refer to the project documentation or create an issue in the repository.