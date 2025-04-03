# RAG_diversity

This repository contains code for the project of enhancing diversity of RAG in LLM through diversity-aware retrieval. 


## Usage Guidelines

### Application of RAG Pipelines to Different Datasets

Only "threshold.ipynb", "top-k.ipynb", "top-m.ipynb" and "top-p.ipynb" need to be run. In each of these files, two parts are contained, for normal RAG pipeline and for evaluation respectively.

The normal RAG pipeline integrate all different processes into a chain. You are welcome to delete the comment signs and try running them.

The part for evaluation seperates different processes, but achieves the same function. This is because evaluation on a dataset for a specific parameter needs to keep other parameters unchanged. As a result, some retrieval processes may be same for different values of the parameter studied. To save the computation time and accelerate the evaluation efficiency, the full chain is decomposed in such a way.

As for the datasets, they are not uploaded due to the limitation of file sizes. However, by running the codes, you will obtain the same datastes locally.

The generated answers on datasets are stored in "top-k_results" and "threshold_results".

### Evaluation of Question-answer Pairs

Since the actual generation of answers on different datasets are parallized by MapReduce. The generated answers of a datasets are stored in several folders. They need to be merged and converted into the required format by "merge_data.ipynb". The generated transformed answers are stored in "clean_data" folder.

"eval-wmd.ipynb" is used to evaluate the answers of different pipelines on different datasets by RougeL and WMD metrics.

"eval_bleu.ipynb" and "eval_tf-idf.ipynb" are used to evaluate them by BLEU and TF-IDF.

The evaluation result details are stored in "wmd", "bleu" and "tf-idf" folders.

Besides, the extracted evaluation summarization are stored in "k-summary.txt" and "th.summary.t]xt".

### About Selection of Datapoints

Respective 1000 indices are randomly choosed from the range of 1800 (the size of the efficient version of NQ), the size of TriviaQA, the size of flattened SQuAD and the size of flattened ASQA sequentially, with the random state seed of 42. These indices are used to locate questions in NQ, TriviaQA, SQuAD and ASQA, respectively. 

Because of the limitation of number of relevant articles in corpus, less than 1000 questions are valid for generating answers in "top-k_results" and "threshold_results". The first 500 question-answer pairs will be used for final evaluation of RougeL and diversity, which should be repeatable.

However, for the top-k method, on TriviaQA, since the first 1000 questions don't contain enough (500) valid questions, 1500 indices are choosed from the same random state and the first 500 valid questions are used for evaluation. For SQuAD and ASQA, they still use the questions indiced by the original way of generating indices.

### Important Notes

The author used a wrong code for similarity threshold retrieval method, and generated some wrong evaluation results, which are stored with top-k results. Sometimes, you open a folder which should contain top-k and threshold results, but you only see threshold results and a folder named or containing "old". The top-k results are in the "old" folder, which are not separated from the old version of threshold results (wrong version). The author didn't separate them in order to keep the most original data. If you found it difficult to separate them by yourself, please contact the author for assistance.

### Warnings

The author didn't modify the comments when modifying the codes. Therefore, sometimes the comments are irrelevant with the code. Please ignore them. 

## Similarity-Threshold Retrieval:

"threshold.ipynb" contains code for studying the effect of similarity-threshold retrieval on the performance of RAG. Fisrt, top-m relevant documents are retrieved. Then, documents are selected with equal probabilities. Only those having similarity scores lower than a threshold with previous selected documents can be kept. Eventually at most k documents will be returned to the LLM together with query.

## Top-k Retrieval

"top-k.ipynb" contains code for top-k retrieval. The method simply returns top-k documents most relevant to the query. The code studies the effect on generation diversity of different k values.

## Top-m Retrieval, Temperature, Noise

In "top-m.ipynb" file.

### Random Sample k from Top-m

The method retrieved top-k most relevant documents, and then randomly sample k documents with certain probabilities. The probabilities are calculated through their similarities with query and normalized.

### Temperature

The parameter "temperature" applies a function to the probabilities. Please refer to the explanation of temperature definition in ChatGPT.

### Noise 

The parameter "std_error" adds noise to the probabilities. The noise follows Gaussian distribution centered at 0 with a user-defined standard error std_error. 

There is no need to add non-zero center to the Guassian Distribution, because the parameter temperature is equivalent with adding -log(temperature) deviation to all probabilities. Please refer to the mathematical expression of temperature definition.

## Top-p Retrieval

"top-p.ipynb" contains code for top-p retrieval, temperature, and noise.

As for the definition of top-p retrieval, please ask ChatGPT for detailed explanation.

## Datasets

Natural Questions, TriviaQA, SQuAD and ASQA are used here. 500 questions randomly sampled from their datasets are used for evaluation. Actually I think evaluation on diversity of short text generation is not as effective as long text generation.

A potential evaluation metric for long-text generation is to ask LLM write long articles and ask LLM to grade them. But since the generation process relies on RAG to get knowledge outside parameters, the LLM for evaluation should be smarter or equiped with RAG, too. For example, ask ChatGPT4 or higher versions to give scores, instead of pure generation models.











