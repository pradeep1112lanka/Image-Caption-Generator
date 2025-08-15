# evaluate.py
import pickle
from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.models import load_model
from predict import generate_desc
from utils import load_tokenizer, load_features

def load_clean_descriptions(filename='cleaned_descriptions.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def evaluate_model(model, descriptions, features, tokenizer, max_len):
    actual, predicted = [], []
    for img_id, refs in descriptions.items():
        if img_id not in features:
            continue
        photo = features[img_id]
        yhat = generate_desc(model, tokenizer, photo, max_len)
        yhat = yhat.replace('startseq', '').replace('endseq', '').strip().split()
        predicted.append(yhat)
        actual.append([r.replace('startseq', '').replace('endseq', '').strip().split() for r in refs])
    bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu1, bleu2, bleu3, bleu4

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='image_caption_model.h5')
    parser.add_argument('--features', default='features.pkl')
    parser.add_argument('--descriptions', default='cleaned_descriptions.pkl')
    parser.add_argument('--tokenizer', default='tokenizer.pkl')
    args = parser.parse_args()

    model = load_model(args.model)
    features = load_features(args.features)
    descriptions = load_clean_descriptions(args.descriptions)
    tokenizer = load_tokenizer(args.tokenizer)
    # compute max_len from descriptions
    max_len = max(len(d.split()) for refs in descriptions.values() for d in refs)
    b1, b2, b3, b4 = evaluate_model(model, descriptions, features, tokenizer, max_len)
    print(f"BLEU-1: {b1:.4f} BLEU-2: {b2:.4f} BLEU-3: {b3:.4f} BLEU-4: {b4:.4f}")
