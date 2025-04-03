import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer
import gensim.downloader as api
from gensim.models import KeyedVectors
import spacy
import os

class EvaluationMetrics():
    def __init__(self):

        self.vectorizer = TfidfVectorizer()
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        self.word2vec_path = "word2vec_local.model"
        self.spacy_model_path = "en_custom_model"

        self.model = self.load_or_download_word2vec()
        self.nlp = self.load_or_download_spacy()

    def load_or_download_word2vec(self):
        """Loads the Word2Vec model if available, otherwise downloads and saves it."""
        if os.path.exists(self.word2vec_path):
            print("Loading Word2Vec model from local storage...")
            return KeyedVectors.load(self.word2vec_path)
        else:
            print("Downloading Word2Vec model...")
            model = api.load("word2vec-google-news-300")
            model.save(self.word2vec_path)
            return model

    def load_or_download_spacy(self):
        """Loads spaCy model from local storage if available, otherwise downloads and saves it."""
        if os.path.exists(self.spacy_model_path):
            print("Loading spaCy model from local storage...")
            return spacy.load(self.spacy_model_path)
        else:
            print("Downloading spaCy model...")
            nlp = spacy.load("en_core_web_md")  # Load downloaded model
            nlp.to_disk(self.spacy_model_path)  # Save it locally
            return nlp

    def preprocess_sentence(self, sentence):
        words = sentence.lower().split()  # You can use a more advanced tokenizer
        valid_words = ''.join([word for word in words if word in self.model.key_to_index])
        return valid_words

    def calculate_custom_rougeL(self, reference, generated):
        """
        Calculate the Rouge-L F1 score between a reference and a generated answer.
        """
        if reference in generated:
            return 1.0
        
        scores = self.scorer.score(reference, generated)
        return scores["rougeL"].fmeasure

    def calculate_f1(self, candidates, references):
        """
        Evaluate the Rouge-1, Rouge-2, and Rouge-L F1 scores for a list of candidate and reference answers.
        """
        rouge1_f1_sum = 0
        rouge2_f1_sum = 0
        rougeL_f1_sum = 0
        num_samples = len(candidates)
    
        for can_list, ref_list in zip(candidates, references):
            scores = [[self.scorer.score(ref, can) for ref in ref_list] for can in can_list]
            aggregated_scores = {
                "rouge1_f1": np.mean([max([s['rouge1'].fmeasure for s in score]) for score in scores]),
                "rouge2_f1": np.mean([max([s['rouge2'].fmeasure for s in score]) for score in scores]),
                "rougeL_f1": np.mean([max(self.calculate_custom_rougeL(ref, can) for ref in ref_list) for can in can_list]),
            }
            
            rouge1_f1_sum += aggregated_scores["rouge1_f1"]
            rouge2_f1_sum += aggregated_scores["rouge2_f1"]
            rougeL_f1_sum += aggregated_scores["rougeL_f1"]
        
        avg_rouge1_f1 = rouge1_f1_sum / num_samples
        avg_rouge2_f1 = rouge2_f1_sum / num_samples
        avg_rougeL_f1 = rougeL_f1_sum / num_samples

        return (avg_rouge1_f1, avg_rouge2_f1, avg_rougeL_f1)

    def calculate_inner_sentence_diversity(self, sentence):
        """
        Calculate inner-sentence diversity using Word Mover's Distance (WMD)
        """
        doc = self.nlp(sentence)
        words = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
        
        if len(words) == 0:
            return 0.0
        
        total_wmd = 0.0
        count = 0
        
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                word1 = words[i]
                word2 = words[j]
                
                if word1 in self.model and word2 in self.model:
                    total_wmd += self.model.wmdistance(word1, word2)
                    count += 1
        
        return total_wmd / count if count > 0 else 0.0

    def calculate_inter_sentence_diversity(self, sentences):
        """
        Calculate inter-sentence diversity using Word Mover's Distance (WMD)
        """
        total_wmd = 0.0
        count = 0
        
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                sentence1 = sentences[i]
                sentence2 = sentences[j]

                if not sentence1 or not sentence2:
                    continue

                wmd_score = self.model.wmdistance(sentence1, sentence2)
                
                if np.isfinite(wmd_score):
                    total_wmd += wmd_score
                    count += 1
                else:
                    print(f"Warning: WMD returned inf for '{sentence1}' and '{sentence2}'")
        
        return total_wmd / count if count > 0 else 0.0

    def calculate_diversities(self, answers):
        """
        Calculate diversity scores (inner and inter sentence diversity) for answers.
        """
        inner_diversities = []
        inter_diversities = []
        
        for answer_set in answers:
            answer_set = [self.preprocess_sentence(answer) for answer in answer_set]
            answer_set = [answer for answer in answer_set if answer]
            inner_diversity_scores = [self.calculate_inner_sentence_diversity(answer) for answer in answer_set]
            inner_diversity_mean = np.mean(inner_diversity_scores) if inner_diversity_scores else 0.0
            inner_diversities.append(inner_diversity_mean)
            
            inter_diversity = self.calculate_inter_sentence_diversity(answer_set)
            inter_diversities.append(inter_diversity)
        
        inner_diversity_mean = np.mean(inner_diversities)
        inter_diversity_mean = np.mean(inter_diversities)
        
        return inner_diversity_mean, inter_diversity_mean

    def cal_scores(self, cans, refs):
        # Create separate lists for each score type
        rouge1_f1_list = []
        rouge2_f1_list = []
        rougeL_f1_list = []
        inner_diversity_list = []
        inter_diversity_list = []

        for can_list in cans:
            # Calculate Rouge scores
            rouge1_f1, rouge2_f1, rougeL_f1 = self.calculate_f1(can_list, refs)
            
            # Calculate diversity scores
            inner_diversity, inter_diversity = self.calculate_diversities(can_list)

            # Append the results to the corresponding lists
            rouge1_f1_list.append(rouge1_f1)
            rouge2_f1_list.append(rouge2_f1)
            rougeL_f1_list.append(rougeL_f1)
            inner_diversity_list.append(inner_diversity)
            inter_diversity_list.append(inter_diversity)

        # Return the lists of scores
        return rouge1_f1_list, rouge2_f1_list, rougeL_f1_list, inner_diversity_list, inter_diversity_list

