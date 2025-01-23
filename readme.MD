# Benchmarking Toxic Symbol Detection in Multimodal Memes

## Dataset and results

Dataset: https://huggingface.co/datasets/jasonjheeta/toxic-symbol-detection-in-memes/tree/main  
Models and results: https://huggingface.co/jasonjheeta/toxic-symbol-detection-in-memes/tree/main

## How to run?

Make sure you use python 3.9 and run `pip install -r requirements.txt` in a virtual environment.

You can run `python run_all_models.py` to replicate the results by running all models. 

If you want to run a individual model (for example a text model), you can use the script called `text_model.py`. You can call the script with distilbert and 56-way classification like this:  
`python text_model.py distilbert toxic-symbols`.