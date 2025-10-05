#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re
from typing import List, Dict, Tuple
import os

class WordEmbeddingProcessor:
    """Xử lý embedding và vector hóa văn bản"""
    
    def __init__(self, embedding_path: str):
        self.embedding_path = embedding_path
        self.embeddings = None
        self.vocab = None
        self.vocab_to_idx = {}
        self.embedding_dim = None
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load embeddings từ file .npz"""
        try:
            data = np.load(self.embedding_path)
            self.embeddings = data['embeddings']
            self.vocab = data['vocab']
            self.vocab_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
            self.embedding_dim = self.embeddings.shape[1]
            print(f"Loaded embeddings: {self.embeddings.shape}, vocab size: {len(self.vocab)}")
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            raise
    
    def preprocess_text(self, text: str) -> List[str]:
        """Tiền xử lý text thành list words"""
        if pd.isna(text) or text is None:
            return []
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [w.strip() for w in text.split() if w.strip()]
    
    def get_word_embedding(self, word: str) -> np.ndarray:
        """Embedding cho 1 từ, trả về zero nếu không có"""
        return self.embeddings[self.vocab_to_idx[word]] if word in self.vocab_to_idx else np.zeros(self.embedding_dim)
    
    def vectorize_text_words(self, text: str) -> Tuple[List[str], List[np.ndarray]]:
        """Vector hóa từng từ trong câu"""
        words = self.preprocess_text(text)
        embeddings = [self.get_word_embedding(w) for w in words]
        return words, embeddings
    
    def get_sentence_embedding(self, text: str) -> np.ndarray:
        """Tạo embedding câu = trung bình các word embeddings"""
        words, embeddings = self.vectorize_text_words(text)
        return np.mean(embeddings, axis=0) if embeddings else np.zeros(self.embedding_dim)


def create_word_analysis_csv(processor, df, output_path="word_analysis_for_comparison.csv"):
    """Tạo CSV phân tích từng từ"""
    rows = []
    for _, row in df.iterrows():
        words = processor.preprocess_text(row['text'])
        for w in words:
            if w in processor.vocab_to_idx:
                emb = processor.embeddings[processor.vocab_to_idx[w]]
                row_data = {'sample_id': row['id'], 'text': row['text'], 'sentiment': row['sentiment'],
                            'word': w, 'status': 'found', 'norm': float(np.linalg.norm(emb))}
                for i in range(min(10, len(emb))):
                    row_data[f'emb_{i}'] = float(emb[i])
            else:
                row_data = {'sample_id': row['id'], 'text': row['text'], 'sentiment': row['sentiment'],
                            'word': w, 'status': 'missing', 'norm': 0.0}
                for i in range(min(10, processor.embedding_dim)):
                    row_data[f'emb_{i}'] = 0.0
            rows.append(row_data)
    df_words = pd.DataFrame(rows)
    df_words.to_csv(output_path, index=False)
    print(f"Saved word analysis to {output_path}")
    return df_words


def create_sentence_comparison_csv(processor, df, output_path="sentence_embeddings_for_comparison.csv"):
    """Tạo CSV embedding câu và thống kê"""
    rows = []
    for _, row in df.iterrows():
        words = processor.preprocess_text(row['text'])
        found, missing = [], []
        embeddings = []
        for w in words:
            if w in processor.vocab_to_idx:
                found.append(w)
                embeddings.append(processor.embeddings[processor.vocab_to_idx[w]])
            else:
                missing.append(w)
        sent_emb = np.mean(embeddings, axis=0) if embeddings else np.zeros(processor.embedding_dim)
        row_data = {
            'sample_id': row['id'], 'text': row['text'], 'sentiment': row['sentiment'],
            'total_words': len(words),
            'found_words': len(found),
            'missing_words': len(missing),
            'coverage_pct': (len(found)/len(words)*100) if words else 0,
            'found_list': ', '.join(found),
            'missing_list': ', '.join(missing),
            'embedding_norm': float(np.linalg.norm(sent_emb))
        }
        for i in range(min(20, len(sent_emb))):
            row_data[f'dim_{i}'] = float(sent_emb[i])
        rows.append(row_data)
    df_sent = pd.DataFrame(rows)
    df_sent.to_csv(output_path, index=False)
    print(f"Saved sentence embeddings to {output_path}")
    return df_sent


def main():
    """Chạy xử lý embedding và tạo CSV cho dataset"""
    embedding_path = "word_embeddings.npz"
    test_data_path = "dataset/dataset/test.csv"

    print("Loading embeddings...")
    processor = WordEmbeddingProcessor(embedding_path)

    print("Loading dataset...")
    df_all = pd.read_csv(test_data_path)
    df = df_all[df_all['sentiment'].isin(['positive', 'negative'])].copy()
    print(f"Dataset: {len(df)} samples (pos+neg)")

    print("\nCreating CSVs...")
    df_words = create_word_analysis_csv(processor, df)
    df_sent = create_sentence_comparison_csv(processor, df)

    # Summary
    print("\nSummary:")
    total_words = len(df_words)
    found_words = len(df_words[df_words['status']=='found'])
    coverage = (found_words/total_words*100) if total_words else 0
    print(f"Total words: {total_words}, Found: {found_words}, Coverage: {coverage:.1f}%")
    for s in ['positive','negative']:
        subset = df_sent[df_sent['sentiment']==s]
        if len(subset):
            print(f"{s}: {len(subset)} samples, avg coverage {subset['coverage_pct'].mean():.1f}%, avg norm {subset['embedding_norm'].mean():.4f}")

    print("\nDone. CSVs ready for direct comparison.")


if __name__ == "__main__":
    main()
