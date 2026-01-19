"""
Evaluation Module
Evaluates model performance using multiple metrics
"""

import os
import json
import yaml
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from bert_score import score
from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np
from collections import Counter


class ModelEvaluator:
    """Evaluates paraphrase model performance"""
    
    def __init__(self, model_path: str, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model from {model_path} on {self.device}...")
        self.default_tone = self.config.get('style', {}).get('tones', ['academic'])[0]
        self.default_strength = self.config.get('style', {}).get('strengths', ['medium'])[0]
        self.tone_tokens = [f"<tone={t}>" for t in self.config.get('style', {}).get('tones', [])]
        self.strength_tokens = [f"<strength={s}>" for s in self.config.get('style', {}).get('strengths', [])]
        self.special_tokens = list({*self.tone_tokens, *self.strength_tokens})
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        base_model_name = self.config['model']['base_model']
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        
        # Check if LoRA model
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            self.model = base_model
        
        # Ensure tokenizer/model know about control tokens
        if self.special_tokens:
            self.tokenizer.add_tokens(self.special_tokens, special_tokens=False)
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.model.to(self.device)
        self.model.eval()
        
        self.max_length = self.config['training']['max_length']
        
        # Initialize scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def generate_paraphrase(self, text: str, tone: str = None, strength: str = None,
                           max_length: int = None, num_beams: int = 4) -> str:
        """Generate paraphrase for input text"""
        if max_length is None:
            max_length = self.max_length
        tone = tone or self.default_tone
        strength = strength or self.default_strength
        
        # Prepare input with control tokens
        input_text = f"paraphrase <tone={tone}> <strength={strength}>: {text}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate with deterministic beam search for evaluation
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                do_sample=False,  # Deterministic for evaluation
                repetition_penalty=1.2,
                length_penalty=1.0,
            )
        
        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated
    
    def calculate_bertscore(self, predictions: list, references: list) -> dict:
        """Calculate BERTScore"""
        print("Calculating BERTScore...")
        
        try:
            P, R, F1 = score(
                predictions,
                references,
                lang='en',
                verbose=True,
                device=str(self.device)
            )
            
            return {
                'precision': float(P.mean().item()),
                'recall': float(R.mean().item()),
                'f1': float(F1.mean().item())
            }
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    def calculate_rouge(self, predictions: list, references: list) -> dict:
        """Calculate ROUGE scores"""
        print("Calculating ROUGE scores...")
        
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in tqdm(zip(predictions, references), total=len(predictions)):
            scores = self.rouge_scorer.score(ref, pred)
            for metric in rouge_scores:
                rouge_scores[metric].append(scores[metric].fmeasure)
        
        return {
            metric: {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores))
            }
            for metric, scores in rouge_scores.items()
        }
    
    def calculate_fluency_metrics(self, texts: list) -> dict:
        """Calculate fluency-related metrics"""
        print("Calculating fluency metrics...")
        
        avg_length = np.mean([len(text.split()) for text in texts])
        
        # Count punctuation errors (simple heuristic)
        punctuation_errors = 0
        for text in texts:
            # Check for common punctuation issues
            if text and text[-1] not in '.!?':
                punctuation_errors += 1
        
        punctuation_error_rate = punctuation_errors / len(texts) if texts else 0
        
        return {
            'average_length': float(avg_length),
            'punctuation_error_rate': float(punctuation_error_rate)
        }
    
    def calculate_diversity(self, texts: list, n: int = 2) -> dict:
        """Calculate diversity using distinct-n metric"""
        print(f"Calculating diversity (distinct-{n})...")
        
        all_ngrams = []
        for text in texts:
            words = text.lower().split()
            ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
            all_ngrams.extend(ngrams)
        
        unique_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)
        
        distinct_n = unique_ngrams / total_ngrams if total_ngrams > 0 else 0
        
        return {
            f'distinct_{n}': float(distinct_n),
            'unique_ngrams': unique_ngrams,
            'total_ngrams': total_ngrams
        }
    
    def evaluate(self, test_file: str, sample_size: int = None):
        """Evaluate model on test set"""
        print(f"\n{'='*60}")
        print("Starting evaluation...")
        print(f"{'='*60}")
        
        # Load test data
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        if sample_size:
            test_data = test_data[:sample_size]
        
        print(f"Evaluating on {len(test_data)} samples...")
        
        # Generate predictions
        predictions = []
        references = []
        inputs = []
        
        print("\nGenerating paraphrases...")
        for item in tqdm(test_data):
            input_text = item['input']
            target_text = item['target']
            
            # Generate paraphrase
            generated = self.generate_paraphrase(input_text)
            
            predictions.append(generated)
            references.append(target_text)
            inputs.append(input_text)
        
        # Calculate metrics
        results = {}
        
        # BERTScore
        if 'bertscore' in self.config['evaluation']['metrics']:
            results['bertscore'] = self.calculate_bertscore(predictions, references)
        
        # ROUGE
        if 'rouge' in self.config['evaluation']['metrics']:
            results['rouge'] = self.calculate_rouge(predictions, references)
        
        # Fluency
        if 'fluency' in self.config['evaluation']['metrics']:
            results['fluency'] = self.calculate_fluency_metrics(predictions)
        
        # Diversity
        if 'diversity' in self.config['evaluation']['metrics']:
            results['diversity'] = self.calculate_diversity(predictions)
        
        # Print results
        print(f"\n{'='*60}")
        print("Evaluation Results:")
        print(f"{'='*60}")
        print(json.dumps(results, indent=2))
        
        # Save results
        results_file = os.path.join(
            self.config['paths']['models_dir'],
            "evaluation_results.json"
        )
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Save sample predictions
        samples_file = os.path.join(
            self.config['paths']['models_dir'],
            "evaluation_samples.json"
        )
        
        samples = [
            {
                'input': inp,
                'reference': ref,
                'prediction': pred
            }
            for inp, ref, pred in zip(inputs[:100], references[:100], predictions[:100])
        ]
        
        with open(samples_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        print(f"Sample predictions saved to: {samples_file}")
        
        return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <model_path> [test_file]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if test_file is None:
        # Use default test file
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        processed_data_dir = config['paths']['processed_data_dir']
        paws_test = os.path.join(processed_data_dir, "paws_test.json")
        if os.path.exists(paws_test):
            test_file = paws_test
        else:
            test_file = os.path.join(processed_data_dir, "combined_raw_test.json")
    
    evaluator = ModelEvaluator(model_path)
    evaluator.evaluate(test_file)

