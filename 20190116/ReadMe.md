## Structure Update
1. Use chinese character instead of word cut by jieba
2. Useword vector trained with Zhihu corpus from https://github.com/Embedding/Chinese-Word-Vectors
3. Cut the number of hidden units in each layer of lstm stucture by half
4. Calculate the number of predicted values which is true, test accuracy, train loss and test loss to tracking problem
   
Recording the number of predicted values which is true, is to make sure my model can make balance between a small amount of positive samples and a large number of negative samples.

5. Set the learning rate to 0.0000000000001 of GradientDescentOptimizer
   
Note that I found train loss and test loss tend to be constant during my training process, may be the learning rate is too large to find the optimal value.

6. Use weighted_cross_entropy_with_logits in stead of mean_squared_error
   
Increase the weight of positive sample prediction errors loss to solve the problem of uneven number of positive and negative samples.

## Result
- Accuracy on the verification set: ±65%
- Accuracy of submittion: 18.04%

## To be improved
- The Memory Error haven't been solved yet.
- ...
