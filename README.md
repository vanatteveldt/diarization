# Diarization with openai Whisper and PyAnnote
Simple script to combine whisper and pyannote

# Installation

This is not currently distributed as a package, so clone the repository, create a virtual environment and install the requirements:

```
git clone https://github.com/vanatteveldt/diarization
cd diarization
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install -r requirements.txt
```

Next, please follow the steps at https://github.com/pyannote/pyannote-audio to accept their license terms and get a huggingface access token. 
Then, place this access token in a .env file:

```
echo "HUGGINGFACE_TOKEN=hf_yourtoken" > .env
```

# Example Usage

Example command line usage:

```
python diarizer.py input.wav > output.csv
```

Or using Dutch language and printing a human readable format:

```
python diarizer.py input.wav --language nl --output md 
```
