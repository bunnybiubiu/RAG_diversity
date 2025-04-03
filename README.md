# RAG_diversity

This repository contains code for the project of enhancing diversity of RAG in LLM through diversity-aware retrieval. 


## Usage Guidelines

Only "threshold.ipynb", "top-k.ipynb", "top-m.ipynb", and "top-p.ipynb" need to be run. In each of these files, two parts are contained, for normal RAG pipeline and for evaluation respectively. 

The normal RAG pipeline integrate all different processes into a chain. You are welcome to delete the comment signs and try running them.

The part for evaluation seperates different processes, but achieves the same function. This is because evaluation on a dataset for a specific parameter needs to keep other parameters unchanged. As a result, some retrieval processes may be same for different values of the parameter studied. To save the computation time and accelerate the evaluation efficiency, the full chain is decomposed in such a way.

### Notes

Sometimes the comments are irrelevant with the code. Please ignore them. I didn't revise comments when I did modifications to the code.

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











