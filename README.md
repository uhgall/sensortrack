# sensortrack

A package for using an AI vision model to extract data from images and plot time series data.

## Installation


```bash
python -m venv venv      
source venv/bin/activate 
pip install -e .             
pip install -r requirements.txt
```

You'll need an OpenRouter API key to use this. Or you can change the code to use a different LLM provider. 

Put it in a .env file in the root directory. Should look like this:

```
OPENROUTER_API_KEY=sk-or-v1-<your-key>
```

## How to run 

```bash
cd examples/co2
python process_images.py test
python analyze_data.py test
```

See examples/co2 for a complete example of using sensortrack to compare CO2 sensors, which is what I wrote this for originally. 
