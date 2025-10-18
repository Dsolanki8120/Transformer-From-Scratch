# Import required libraries
import torch
import torch.nn as nn
from datasets import load_dataset                # For loading text datasets (from Hugging Face)
from tokenizers import Tokenizer                 # For creating / loading tokenizers
from tokenizers.models import WordLevel          # Word-level tokenizer model
from tokenizers.trainers import WordLevelTrainer # Helps train the tokenizer on text
from tokenizers.pre_tokenizers import Whitespace # Splits text into words by spaces
from pathlib import Path                         # For handling file paths
from torch.utils.data import Dataset,DataLoader,random_split

# ------------------------------------------------------------
# Function 1: Generator to yield all sentences in one language
# ------------------------------------------------------------
def get_all_sentences(ds, lang):
    """
    This generator function takes a dataset (ds) and a language code (lang),
    and yields all the sentences in that specific language.
    Used for training the tokenizer.
    
    Example:
      ds: [{'translation': {'en': 'Hello', 'fr': 'Bonjour'}}]
      lang: 'en'
      -> yields 'Hello'
    """
    for item in ds:
        yield item['translation'][lang]

# ------------------------------------------------------------
# Function 2: Build or load tokenizer for a given language
# ------------------------------------------------------------
def get_or_build_tokenizer(config, ds, lang):
    """
    If a tokenizer already exists (saved as JSON), load it from disk.
    Otherwise, build a new tokenizer from the dataset and save it.
    
    Arguments:
      config : configuration dictionary with tokenizer file paths
      ds     : dataset containing parallel text (source + target)
      lang   : which language to build tokenizer for (e.g., 'en', 'fr')
    """
    # Build the tokenizer file path (example: "../tokenizers/tokenizer_en.json")
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    
    # If tokenizer file does not exist, we train a new one
    if not Path.exists(tokenizer_path):
        print(f"🔧 Building new tokenizer for language: {lang}")

        # 1️ Initialize empty WordLevel tokenizer (unknown token = [UNK])
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))

        # 2️ Tell tokenizer to split text by whitespace
        tokenizer.pre_tokenizer = Whitespace()

        # 3️ Define training parameters (special tokens, frequency cutoff)
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2   # ignore rare words
        )

        # 4️ Train tokenizer using the text generator (sentences of that language)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)

        # 5️ Save tokenizer to disk (JSON format)
        tokenizer.save(str(tokenizer_path))
        print(f" Saved tokenizer to {tokenizer_path}")

    else:
        # If tokenizer already exists, load it instead of retraining
        print(f" Loading existing tokenizer for language: {lang}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer


# ------------------------------------------------------------
# Function 3: Load dataset and prepare both source + target tokenizers
# ------------------------------------------------------------
def get_ds(config):
    """
    Loads an English–Hindi dataset, builds or loads both tokenizers,
    and splits the dataset into training and validation subsets.
    """
    #  1. Load parallel English–Hindi dataset from Hugging Face
    # opus100 supports many language pairs including en-hi
    ds_raw = load_dataset("opus100", f"{config['lang_src']}-{config['lang_tgt']}", split="train")

    #  2. Build or load the source (English) tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])

    #  3. Build or load the target (Hindi) tokenizer
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    #  4. Split dataset into 90% training, 10% validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    # random_split expects lengths, not variable names
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    #  5. Return everything needed for next steps
    return train_ds_raw, val_ds_raw, tokenizer_src, tokenizer_tgt
