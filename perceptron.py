import random
import numpy as np
import math
import csv
import pickle
from collections import defaultdict

class Perceptron:
    """
    Implements a multiclass averaged perceptron classifier
    with temperature scaling for calibration.
    """
    
    def __init__(self):
        """
        Initializes an empty Perceptron model.
        """
        print("Initialized empty Perceptron model.")
        
        self.weights = None
        self.feat_map = defaultdict(lambda: len(self.feat_map))
        self.class_map = defaultdict(lambda: len(self.class_map))
        self.idx_to_class = {}
        self.bias_id = 0
        self.n_classes = 0
        self.n_features = 0
        self.temperature = 1.0

    def train(self, train_data, dev_data, num_iterations=10):
        """
        Trains the perceptron model on the provided data.
        """
        print(f"Starting training with {num_iterations} iterations...")

        print("Preprocessing data...")
        self.bias_id = self.feat_map['__BIAS__']
        
        tr_samples = []
        for label, tokens in train_data:
            label_id = self.class_map[label]
            feat_ids = [self.feat_map[t] for t in tokens] + [self.bias_id]
            tr_samples.append((label_id, np.array(feat_ids, dtype=np.int32)))

        dev_samples = []
        for label, tokens in dev_data:
            label_id = self.class_map[label] 
            feat_ids = [self.feat_map[t] for t in tokens if t in self.feat_map] + [self.bias_id]
            dev_samples.append((label_id, np.array(feat_ids, dtype=np.int32)))

        self.n_classes = len(self.class_map)
        self.n_features = len(self.feat_map)
        self.idx_to_class = {i: c for c, i in self.class_map.items()}
        
        print(f"Done. Found {self.n_classes} classes and {self.n_features} features.")

        w = np.zeros((self.n_classes, self.n_features), dtype=np.float64)
        wa = np.zeros((self.n_classes, self.n_features), dtype=np.float64)
        avgc = 1.0
        best_dev_acc = -1.0 
        
        for it in range(num_iterations):
            print(f"Iteration {it}")
            random.shuffle(tr_samples)
            
            for label_id, feat_ids in tr_samples:
                scores = np.sum(w[:, feat_ids], axis=1)
                pred_id = np.argmax(scores)

                if pred_id != label_id:
                    w[pred_id, feat_ids] -= 1
                    wa[pred_id, feat_ids] -= avgc
                    w[label_id, feat_ids] += 1
                    wa[label_id, feat_ids] += avgc

            avgc += 1.0

            w_avg = w - (wa / avgc)
            
            corr = 0
            for label_id, feat_ids in dev_samples:
                scores = np.sum(w_avg[:, feat_ids], axis=1)
                pred_id = np.argmax(scores)
                if pred_id == label_id:
                    corr += 1
            
            dev_acc = corr / len(dev_samples)
            print(f"  Dev Accuracy: {corr} / {len(dev_samples)} = {dev_acc:.3f}")

            if dev_acc > best_dev_acc:
                self.weights = w_avg
                best_dev_acc = dev_acc
                print(f"  New best dev accuracy. Weights saved.")
        
        if self.weights is None:
            self.weights = w - (wa / avgc)
            
        print(f"Best dev accuracy: {best_dev_acc:.3f}")

    def tune_temperature(self, dev_data):
        """
        Finds the optimal temperature T for softmax calibration
        in a trained model by minimizing NLL on the dev set.
        """
        if self.weights is None:
            print("Error: Model must be trained before tuning temperature.")
            return

        print("\nTuning temperature on dev set...")
        
        dev_scores_and_labels = []
        for label, tokens in dev_data:
            if label not in self.class_map:
                continue
            label_id = self.class_map[label]
            raw_scores = self._get_scores(tokens)
            dev_scores_and_labels.append((raw_scores, label_id))

        def calculate_nll(T):
            total_nll = 0.0
            for scores, label_id in dev_scores_and_labels:
                scaled_scores = scores / T
                
                probs = np.exp(scaled_scores - np.max(scaled_scores))
                probs /= np.sum(probs)
                
                correct_prob = probs[label_id] + 1e-9 
                
                total_nll += -np.log(correct_prob)
                
            return total_nll / len(dev_scores_and_labels) # Average NLL

        best_T = 1.0
        best_nll = calculate_nll(1.0)
        print(f"  Initial NLL (T=1.0): {best_nll:.4f}")

        for T in np.linspace(0.5, 4.0, 50):
            nll = calculate_nll(T)
            if nll < best_nll:
                best_nll = nll
                best_T = T
        
        self.temperature = best_T
        print(f"  Tuning complete. Best T={self.temperature:.3f} (NLL={best_nll:.4f})")

    def save_model(self, filepath):
        """
        Saves the trained model (weights, maps, and temperature) to a file.
        """
        print(f"Saving model to {filepath}...")
        
        model_data = {
            'weights': self.weights,
            'feat_map': dict(self.feat_map),
            'class_map': dict(self.class_map),
            'idx_to_class': self.idx_to_class,
            'bias_id': self.bias_id,
            'temperature': self.temperature
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print("Model saved successfully.")
        except Exception as e:
            print(f"Error saving model: {e}")

    @classmethod
    def load_model(cls, filepath):
        """
        Loads a trained model from a file.
        """
        print(f"Loading model from {filepath}...")
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
            model = cls()
            model.weights = model_data['weights']
            model.feat_map.update(model_data['feat_map'])
            model.class_map.update(model_data['class_map'])
            model.idx_to_class = model_data['idx_to_class']
            model.bias_id = model_data['bias_id']
            
            model.temperature = model_data.get('temperature', 1.0)
            
            model.n_classes = len(model.class_map)
            model.n_features = len(model.feat_map)
            
            print("Model loaded successfully.")
            return model
            
        except FileNotFoundError:
            print(f"Error: Model file not found at {filepath}")
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def _softmax(self, scores):
        """Applies tempered softmax for normalization."""
        
        scaled_scores = scores / self.temperature
        
        e_scores = np.exp(scaled_scores - np.max(scaled_scores)) 
        return e_scores / e_scores.sum()

    def _get_scores(self, tokens):
        """Converts tokens to feat_ids and calculates raw scores (logits)."""
        feat_ids = [self.feat_map[t] for t in tokens if t in self.feat_map]
        feat_ids.append(self.bias_id)
        
        feat_ids_np = np.array(feat_ids, dtype=np.int32)
        
        if feat_ids_np.size == 0:
            return np.zeros(self.n_classes, dtype=np.float64)
            
        scores = np.sum(self.weights[:, feat_ids_np], axis=1)
        return scores

    def classify_sample(self, tokens, normalize_scores=False):
        """
        Classifies a single sample (list of tokens).
        
        Args:
            tokens (list): A list of feature strings.
            normalize_scores (bool): If True, apply tempered softmax.
            
        Returns:
            list: A list of (label, score) tuples, sorted by score descending.
        """
        if self.weights is None:
            print("Error: Model is not trained. Call .train() first.")
            return []
            
        scores = self._get_scores(tokens)
        
        if normalize_scores:
            scores = self._softmax(scores)
            
        scored_labels = []
        for class_id, score in enumerate(scores):
            label = self.idx_to_class[class_id]
            scored_labels.append((label, score))
            
        scored_labels.sort(key=lambda x: x[1], reverse=True)
        return scored_labels

    def classify_file(self, filepath):
        """
        Classifies samples from a CSV file.
        File format: label,"feat1 feat2 feat3..."
        
        Returns:
            list: A list of predicted label strings.
        """
        if self.weights is None:
            print("Error: Model is not trained. Call .train() first.")
            return []
            
        print(f"\nClassifying file: {filepath}...")
        predictions = []
        correct = 0
        total = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                
                for row in reader:
                    if not row or len(row) < 2: continue
                    
                    true_label = row[0]
                    tokens = row[1].split()
                    
                    scores = self._get_scores(tokens)
                    pred_id = np.argmax(scores)
                    pred_label = self.idx_to_class[pred_id]
                    
                    predictions.append(pred_label)
                    
                    if pred_label == true_label:
                        correct += 1
                    total += 1
                        
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return []
        
        if total > 0:
            accuracy = correct / total
            print(f"Accuracy: {correct} / {total} = {accuracy:.3f}")
        else:
            print("No valid samples found in file.")
            
        return predictions

