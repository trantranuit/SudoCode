#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re
import json
import sys
import os
from typing import List, Tuple, Dict, Any

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Thêm embedding_evaluation vào path
sys.path.append(os.path.join(os.path.dirname(__file__), 'embedding_evaluation', 'embedding_evaluation'))

# Import từ embedding evaluation framework
try:
    from scipy.stats import spearmanr
    EMBEDDING_EVAL_AVAILABLE = True
    print("Basic evaluation components available")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'embedding_evaluation', 'embedding_evaluation'))
        from evaluate_similarity import EvaluationSimilarity  
        from evaluate_concreteness import EvaluationConcreteness
        REPO_EVAL_AVAILABLE = True
        print("Successfully imported embedding evaluation repo framework")
    except ImportError as e:
        REPO_EVAL_AVAILABLE = False
        print(f"Repo evaluation not available: {e}")
        print("Will use basic evaluation methods instead")
        
except ImportError as e2:
    print(f"Error: Basic evaluation not available: {e2}")
    EMBEDDING_EVAL_AVAILABLE = False
    REPO_EVAL_AVAILABLE = False


class SentenceEmbeddingComparison:
    """So sánh sentence embeddings bằng trung bình cộng word vectors"""
    
    def __init__(self, embedding_path: str):
        self.embedding_path = embedding_path
        self.embeddings = None
        self.vocab = None
        self.vocab_to_idx = {}
        self.embedding_dict = {}
        self.embedding_dim = None
        
        self.similarity_evaluator = None
        self.concreteness_evaluator = None
        
        self._load_embeddings()
        self._setup_evaluators()
    
    def _load_embeddings(self):
        """Load word embeddings từ file .npz"""
        print("Loading embeddings...")
        data = np.load(self.embedding_path)
        self.embeddings = data['embeddings']
        self.vocab = data['vocab']
        self.vocab_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.embedding_dim = self.embeddings.shape[1]
        self.embedding_dict = {word: self.embeddings[idx] for word, idx in self.vocab_to_idx.items()}
        
        print(f"Loaded embeddings: {self.embeddings.shape}")
        print(f"Vocabulary size: {len(self.vocab)} words")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def _setup_evaluators(self):
        """Setup evaluation components nếu repo có sẵn"""
        if EMBEDDING_EVAL_AVAILABLE and REPO_EVAL_AVAILABLE:
            try:
                self.similarity_evaluator = EvaluationSimilarity()
                self.concreteness_evaluator = EvaluationConcreteness()
                print("Initialized embedding evaluation framework from repo")
            except Exception as e:
                print(f"Warning: Could not initialize repo evaluators: {e}")
                self.similarity_evaluator = None
                self.concreteness_evaluator = None
        else:
            print("Embedding evaluation repo framework not available - using basic methods")
            self.similarity_evaluator = None
            self.concreteness_evaluator = None
    
    def preprocess_text(self, text: str) -> List[str]:
        """Tiền xử lý text thành list các words"""
        if pd.isna(text) or text is None:
            return []
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [word.strip() for word in text.split() if word.strip()]
    
    def get_sentence_embedding(self, text: str) -> Tuple[np.ndarray, List[str], List[str]]:
        """Tạo sentence embedding bằng trung bình cộng các word vectors"""
        words = self.preprocess_text(text)
        if not words:
            return np.zeros(self.embedding_dim), [], words
        
        word_embeddings = []
        found_words = []
        missing_words = []
        
        for word in words:
            if word in self.vocab_to_idx:
                word_embeddings.append(self.embeddings[self.vocab_to_idx[word]])
                found_words.append(word)
            else:
                missing_words.append(word)
        
        sentence_embedding = np.mean(word_embeddings, axis=0) if word_embeddings else np.zeros(self.embedding_dim)
        return sentence_embedding, found_words, missing_words
    
    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)
    
    def compare_two_sentences(self, sentence1: str, sentiment1: str, 
                              sentence2: str, sentiment2: str) -> Dict[str, Any]:
        """So sánh hai câu và phân tích chi tiết"""
        
        print("="*80)
        print("SENTENCE EMBEDDING COMPARISON")
        print("="*80)
        
        print(f"\nSentence 1 ({sentiment1}): \"{sentence1}\"")
        print(f"\nSentence 2 ({sentiment2}): \"{sentence2}\"")
        
        emb1, found1, missing1 = self.get_sentence_embedding(sentence1)
        emb2, found2, missing2 = self.get_sentence_embedding(sentence2)
        
        coverage1 = len(found1) / (len(found1) + len(missing1)) * 100 if (len(found1) + len(missing1)) > 0 else 0
        coverage2 = len(found2) / (len(found2) + len(missing2)) * 100 if (len(found2) + len(missing2)) > 0 else 0
        
        similarity = self.cosine_similarity(emb1, emb2)
        
        if similarity > 0.95:
            interpretation = "Very High similarity"
        elif similarity > 0.90:
            interpretation = "High similarity"
        elif similarity > 0.80:
            interpretation = "Moderate similarity"
        elif similarity > 0.60:
            interpretation = "Low similarity"
        else:
            interpretation = "Very Low similarity"
        
        common_words = set(found1) & set(found2)
        unique1 = set(found1) - set(found2)
        unique2 = set(found2) - set(found1)
        
        result = {
            'sentence1': {
                'text': sentence1, 
                'sentiment': sentiment1, 
                'found_words': found1, 
                'missing_words': missing1, 
                'coverage': float(coverage1)
            },
            'sentence2': {
                'text': sentence2, 
                'sentiment': sentiment2, 
                'found_words': found2, 
                'missing_words': missing2, 
                'coverage': float(coverage2)
            },
            'comparison': {
                'cosine_similarity': float(similarity), 
                'interpretation': interpretation, 
                'common_words': list(common_words), 
                'unique_to_sentence1': list(unique1), 
                'unique_to_sentence2': list(unique2)
            }
        }
        return result
    
    def compare_positive_vs_negative(self, pos_sentence: str, neg_sentence: str) -> Dict[str, Any]:
        """So sánh hai câu positive và negative"""
        pos_emb, _, _ = self.get_sentence_embedding(pos_sentence)
        neg_emb, _, _ = self.get_sentence_embedding(neg_sentence)
        
        sim_pos_neg = self.cosine_similarity(pos_emb, neg_emb)
        
        if sim_pos_neg > 0.90:
            quality = "Poor - Embeddings không phân biệt tốt positive vs negative"
        elif sim_pos_neg > 0.75:
            quality = "Moderate - Có thể phân biệt một phần positive vs negative"
        elif sim_pos_neg > 0.60:
            quality = "Good - Phân biệt tốt positive vs negative"
        else:
            quality = "Excellent - Phân biệt rất tốt positive vs negative"
        
        return {
            'sentences': {'positive': pos_sentence, 'negative': neg_sentence},
            'similarity': float(sim_pos_neg),
            'quality_assessment': quality
        }

def demo_positive_vs_negative_from_data():
    """Demo so sánh câu positive và negative từ dataset"""
    print("="*80)
    print("🎯 DEMO: POSITIVE vs NEGATIVE COMPARISON FROM DATASET")
    print("="*80)
    
    tool = SentenceEmbeddingComparison("word_embeddings.npz")
    
    data_path = "dataset/dataset/test.csv"
    try:
        df = pd.read_csv(data_path)
        
        # Filter chỉ lấy positive và negative (bỏ neutral)
        df_filtered = df[df['sentiment'].isin(['positive', 'negative'])].copy()
        print(f"📂 Loaded dataset: {len(df)} total samples")
        print(f"📂 Filtered dataset: {len(df_filtered)} samples (positive + negative only)")
        
        positive_samples = df_filtered[df_filtered['sentiment'] == 'positive']
        negative_samples = df_filtered[df_filtered['sentiment'] == 'negative']
        
        print(f"✅ Found {len(positive_samples)} positive samples")
        print(f"✅ Found {len(negative_samples)} negative samples")
        
        np.random.seed(42)
        n_samples = 5
        pos_sample_idx = np.random.choice(len(positive_samples), min(n_samples, len(positive_samples)), replace=False)
        neg_sample_idx = np.random.choice(len(negative_samples), min(n_samples, len(negative_samples)), replace=False)
        selected_positive = positive_samples.iloc[pos_sample_idx].reset_index(drop=True)
        selected_negative = negative_samples.iloc[neg_sample_idx].reset_index(drop=True)
        
        print(f"\n📝 SELECTED POSITIVE SENTENCES:")
        for i, (_, row) in enumerate(selected_positive.iterrows()):
            print(f"  {i+1}. \"{row['text'][:100]}...\"")
        
        print(f"\n📝 SELECTED NEGATIVE SENTENCES:")
        for i, (_, row) in enumerate(selected_negative.iterrows()):
            print(f"  {i+1}. \"{row['text'][:100]}...\"")
        
        comparisons = []
        print(f"\n🔄 DETAILED COMPARISONS:")
        for i, (_, pos_row) in enumerate(selected_positive.iterrows()):
            for j, (_, neg_row) in enumerate(selected_negative.iterrows()):
                print(f"\n" + "-"*60)
                print(f"COMPARISON {i+1}.{j+1}: Positive vs Negative")
                result = tool.compare_two_sentences(pos_row['text'], 'positive', neg_row['text'], 'negative')
                comparisons.append({'positive_id': pos_row['id'], 'negative_id': neg_row['id'], 'comparison_result': result})
        
        # Summary statistics
        similarities = [comp['comparison_result']['comparison']['cosine_similarity'] for comp in comparisons]
        avg_similarity = np.mean(similarities)
        min_similarity = np.min(similarities)
        max_similarity = np.max(similarities)
        std_similarity = np.std(similarities)
        
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"  Total comparisons: {len(similarities)}")
        print(f"  Average similarity: {avg_similarity:.4f}")
        print(f"  Min similarity: {min_similarity:.4f}")
        print(f"  Max similarity: {max_similarity:.4f}")
        print(f"  Standard deviation: {std_similarity:.4f}")
        
        # Assessment based on similarity
        if avg_similarity > 0.90:
            assessment = "❌ Poor - Embeddings không phân biệt tốt positive vs negative"
        elif avg_similarity > 0.75:
            assessment = "⚠️ Moderate - Có thể phân biệt một phần positive vs negative"
        elif avg_similarity > 0.60:
            assessment = "✅ Good - Phân biệt tốt positive vs negative"  
        else:
            assessment = "🎯 Excellent - Phân biệt rất tốt positive vs negative"
        
        print(f"  Assessment: {assessment}")
        
        # Create comprehensive results (only positive vs negative)
        dataset_comparison_results = {
            'methodology': 'Compare positive vs negative sentences from dataset using averaged word embeddings',
            'dataset_info': {
                'total_samples': len(df),
                'filtered_samples': len(df_filtered),  
                'positive_samples': len(positive_samples),
                'negative_samples': len(negative_samples),
                'samples_used': n_samples
            },
            'selected_sentences': {
                'positive': [{'id': row['id'], 'text': row['text']} for _, row in selected_positive.iterrows()],
                'negative': [{'id': row['id'], 'text': row['text']} for _, row in selected_negative.iterrows()]
            },
            'all_comparisons': comparisons,
            'statistics': {
                'avg_similarity': float(avg_similarity),
                'min_similarity': float(min_similarity),
                'max_similarity': float(max_similarity),
                'std_similarity': float(std_similarity)
            },
            'assessment': assessment
        }
        
        # Save results
        with open('positive_vs_negative_comparison.json', 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(dataset_comparison_results), f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results saved to 'positive_vs_negative_comparison.json'")
        return dataset_comparison_results
        
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {data_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None


if __name__ == "__main__":
    demo_positive_vs_negative_from_data()
