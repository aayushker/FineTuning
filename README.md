# FineTuning

This project demonstrates how to fine-tune a GPT-2 model for sequence classification using the Hugging Face Transformers library.

## Project Structure

- `initializeBaseModel.py`: Initializes the GPT-2 model for sequence classification.
- `loadDataset.py`: Loads the dataset for training and evaluation.
- `tokenizer.py`: Tokenizes the dataset using the GPT-2 tokenizer.
- `evalutation.py`: Defines the metrics for evaluating the model.
- `fineTuner.py`: Contains the `fineTuner` class that sets up the training arguments and trainer.
- `main.py`: Integrates all components and runs the training process.

## Requirements

Install the required packages using `pip`:

```sh
pip install -r requirements.txt
```

## Usage

Run the `main.py` script to fine-tune the GPT-2 model:

```sh
python main.py
```