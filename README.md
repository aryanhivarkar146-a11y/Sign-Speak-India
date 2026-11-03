# SignSpeak India

SignSpeak is a browser-based Indian Sign Language recognition dashboard. It captures camera input, extracts pose and hand landmarks with MediaPipe, and streams them to a FastAPI backend. A trained Keras model predicts the sign and returns confidence scores and alternatives. The project also includes model training and dataset preparation tools.

## Components

- `index.html`: camera-enabled browser dashboard and landmark extraction.
- `backend/`: FastAPI WebSocket service and ISL500 dataset preparation utilities.
- `Aiml developer/`: model inference, training scripts, metadata, and model artifacts.

See [backend/README.md](backend/README.md) for setup and API details.

## License

The MIT license in [LICENSE](LICENSE) covers original source code authored for this project. It does not license the ISL500-derived dataset files or trained model weights. The [ISL500 dataset card](https://huggingface.co/datasets/ISL500/ISL-DATA) says the dataset is intended for research and academic use and asks users to contact its authors about commercial use. Check the relevant terms before using or redistributing those materials.
