# Data
## 0 Description
This folder contains training data.
All SGD/SGD-X data are formated according to the [d3st paper](https://arxiv.org/abs/2201.08904), by using 
an argument index (e.g., s2) to replace an argument's name (e.g., "ticketClass") 
and a value index (e.g., s2.1) to replace a value's name (e.g., "economy").

## 1. Datasets
We include multiple datasets used in this work, including
- `sgd_*`: contains SGD training and validation splits. See
    below about how to preprocess it from the raw SGD data.
    - `sgd_d3st_prompt` saves the API documentation as a single paragraph.
  It is used to train a LLaMA baseline.
    - `sgd_d3st_prompt_jsonInstruct` saves the API documentation in a json object.
  It is used to train our HD-Gist models.
- `sgd_x_*`: contains SGD-X v1-v5 training and validation splits. 
  - `sgd_x_d3st_prompt` saves the API documentation as a single paragraph.
  It is used to evaluate a LLaMA baseline trained on `sgd_d3st_prompt`.
  - `sgd_x_d3st_prompt_jsonInstruct` saves the API documentation in a json object.
  It is used to evaluate our HD-Gist models trained on `sgd_d3st_prompt_jsonInstruct`. 

In this work, we never use any SGD-X training data.
We only use their validation/test set as an out-of-distribution test
for our models trained on SGD. 
See below about how to preprocess it from the raw SGD data.
