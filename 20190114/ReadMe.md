## Structure
1. Use jieba for Chinese word segmentation
2. Use word vector trained with BaiduBaike corpus from https://github.com/Embedding/Chinese-Word-Vectors
3. Align every sentence to max sentence length. Fill the missing length with zeros.
4. 2 3-layered-lstm stuctures for featuring 2 input sentences
5. Compute cosine similarity of the last time-step outputs of the two lstm stucture
6. Using sigmoid function to transform cosine similarity(value from -1 to 1) to a positive number 
7. Rounded up the positive number as the predict value

## Result
- Accuracy on the verification set: ±75%
- Accuracy of submittion: 0

## To be improved
- Segmentation of words is not accurate:

Many words such as "花呗" "借呗" are not recognized as words and are cut apart.

- Word Vectors are not comprehensive:

Many word such as "网商贷" "提额" are not in the dictionary's keys.

- Division of verification set:

I use a part of training data as verification set, and all remaining data to train
