import logging
import torch
from transformers import RobertaTokenizer

from src.model import VADRobertaModel


def log_info(message):
    logging.info(message)

def log_warning(message):
    logging.warning(message)

def log_error(message):
    logging.error(message)

def smoke_test(model_name='roberta-base', max_len=64):
    try:
        logging.info("Starting smoke test...")
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = VADRobertaModel(model_name)
        logging.info("Tokenizer and model loaded successfully.")

        sample_text = "happy"
        encoding = tokenizer.encode_plus(
            sample_text,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        logging.info("Sample text encoded.")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            prediction = outputs['logits'].cpu().numpy()
        logging.info(f"Model prediction: {prediction}")

        logging.info("Smoke test completed successfully.")
        return True
    except Exception as e:
        logging.error(f"Smoke test failed: {e}")
        return False