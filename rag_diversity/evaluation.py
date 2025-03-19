import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

class EvaluationMetrics():
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    
    def calculate_diversity_for_answers(self, answers):
        """
        Calculate the diversity score for a list of answers based on cosine similarity.

        Parameters:
        - answers: List of answers (strings) to compare.

        Returns:
        - diversity_score: A value indicating the diversity (lower is more diverse).
        """
        # Convert the answers into TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(answers)
        
        # Calculate cosine similarity between the answers
        cosine_sim = cosine_similarity(tfidf_matrix)
        
        # Set diagonal values to 0 because we don't want to compare the answer to itself
        np.fill_diagonal(cosine_sim, 0)
        
        # Calculate the mean similarity score for each pair
        mean_similarity = cosine_sim.mean()
        
        # The diversity score is the inverse of the mean similarity
        diversity_score = 1 / (1 + mean_similarity)  # Add 1 to avoid division by 0

        return diversity_score

    
    def eval_diversity(self, all_answers):
        """
        Calculate the mean diversity score for multiple sets of answers.

        Parameters:
        - all_answers: A list of lists, each containing a set of answers.

        Returns:
        - mean_diversity_score: The average diversity score across all answer sets.
        """
        diversity_scores = [self.calculate_diversity_for_answers(answers) for answers in all_answers]
        mean_diversity_score = np.mean(diversity_scores)
        return mean_diversity_score

    
    def calculate_custom_rougeL(self, reference, generated):
        """
        Calculate the Rouge-L F1 score between a reference and a generated answer.

        Parameters:
        - reference: The reference answer (string).
        - generated: The generated answer (string).

        Returns:
        - rougeL_score: The Rouge-L F1 score between the reference and the generated answer.
        """
        # Check if reference is contained in the generated answer
        if reference in generated:
            return 1.0
        
        scores = self.scorer.score(reference, generated)
        return scores["rougeL"].fmeasure

    
    def eval_f1(self, candidates, references):
        """
        Evaluate the Rouge-1, Rouge-2, and Rouge-L F1 scores for a list of candidate and reference answers.

        Parameters:
        - candidates: A list of candidate answers (strings).
        - references: A list of reference answers (strings).

        Returns:
        - avg_rouge1_f1: The average Rouge-1 F1 score.
        - avg_rouge2_f1: The average Rouge-2 F1 score.
        - avg_rougeL_f1: The average Rouge-L F1 score.
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


    
    def cal_scores(self, cans, refs):
        rougeL_scores = list(map(lambda can: self.eval_f1(can, refs)[-1], cans))
        diversity_scores = list(map(lambda can: self.eval_diversity(can), cans))
        scores = [rougeL_scores, diversity_scores]
        return scores

