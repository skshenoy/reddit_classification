# Reddit Classification (trying to spot assholes and bots)

## This project is about pulling Reddit data and trying to classify which subreddit (topic-based forum) it came from, using various supervised learning methods.

## Part 1 was data collection (gathering the Reddit data using the Pushshift API) which can be found in notebook #1.

## Part 2 is the fun stuff, including actual modeling.

### To start off, I decided to compare two advice subreddits with fairly dissimilar content: r/legaladvice and r/relationships. As the names suggest, one is a place “to ask simple legal questions, and to have legal concepts explained” and the other is a place for “helping people and…providing a platform for interpersonal relationship advice”. The general template of each post is the original poster (OP) giving a description of their situation, and others respond in the comments with information and advice. The overall content is different enough that this isn’t a particularly difficult task. I pulled about 2,500 posts from each subreddit, cleaned the text up, and then grid-searched over a few different sets of parameters (the vectorizer used, the set of stop words, the size of the n-grams, number of features, the model itself).

### Overall, the best model was a logistic regression, which performed better with an accuracy score of 97.3% whereas the multinomial naive Bayes had an accuracy score of 93.6%. The vectorizer that was used in the logistic regression model was CountVectorizer() with 500 features, filtering English stop words out, and looking at unigrams and bigrams.