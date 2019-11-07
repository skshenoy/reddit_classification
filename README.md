# Reddit Classification (trying to spot assholes and bots)

## This project is about pulling Reddit data and trying to classify which subreddit (topic-based forum) it came from, using various supervised learning methods.

## Part 1 was data collection (gathering the Reddit data using the Pushshift API) which can be found in [notebook #1](./notebooks/1.%20Collecting%20Reddit%20Posts.ipynb). Part 2 is the fun stuff, including actual modeling.

### Part 2A: Love and Law

To start off, I decided to compare two advice subreddits with fairly dissimilar content: [r/legaladvice](https://www.reddit.com/r/legaladvice) and [r/relationships](https://www.reddit.com/r/relationships). As the names suggest, one is a place “to ask simple legal questions, and to have legal concepts explained” and the other is a place for “helping people and…providing a platform for interpersonal relationship advice”. The general idea of each post is the original poster (OP) giving a description of their situation, and others respond in the comments with information and advice. The overall content is different enough that this isn’t a particularly difficult task. I pulled about 2,500 posts from each subreddit, cleaned the text up, and then grid-searched over a few different sets of parameters (the vectorizer used, the set of stop words, the size of the n-grams, number of features, the model itself).

Overall, the best model was a logistic regression, which performed better with an accuracy score of 97.3% whereas the multinomial naive Bayes had an accuracy score of 93.6%. The vectorizer that was used in the logistic regression model was CountVectorizer() with 500 features, filtering English stop words out, and looking at unigrams and bigrams.

My code and workflow are documented in the 2A notebook.

### Part 2B: Assholery

Moving on to a harder problem: trying to spot assholes! There's an entire subreddit called [r/AmItheAsshole](https://www.reddit.com/r/AmItheAsshole) that describes itself like this:

> A catharsis for the frustrated moral philosopher in all of us, and a place to finally find out if you were wrong in an argument that's been bothering you. Tell us about any non-violent conflict you have experienced; give us both sides of the story, and find out if you're right, or you're the asshole.

Essentially, every post is an OP coming in to describe a (real or fictitious, since people have been known to lie on the Internet) conflict and ideally give both sides of the situation. Commenters ask for more information and then weigh in with their vote, and eventually the moderators' bot will calculate the final verdict. Common votes are YTA (you're the asshole) and NTA (not the asshole), though there are also ESH (everyone sucks here; i.e., they're both assholes) or NAH (no assholes; nobody's at fault). As a sometimes-visitor of this subreddit, I wanted to see if I could train a model to read posts and then decide: *is* OP the asshole?

I ended up creating three different versions of the model:
    - using just the text of OP's post (i.e., modeling _how assholes talk_)
    - using the text of the comments (i.e., modeling *how people talk **about** assholes*)
    - and using both.

Throughout this section, the metric I used to judge the models were the AUC-ROC score and then later the precision score because `asshole` was the positive class and I really wanted to know: could my model find the assholes?
    - Part one: the best model was a multinomial naive Bayes model, with CountVectorizer() (500 features, filtering no stop words out, and looking at unigrams and bigrams). It had a precision of 26%, only a 7.4 improvement over †he 19.5% baseline.
    - Part two: the best model was a logistic regression model, with CountVectorizer() (500 features, filtering no stop words out, and looking at unigrams and bigrams). It had a precision of **55.8%, a 36.3% improvement over †he 19.5% baseline!**
    - Part three: essentially the same as part two, with 1.1% lower precision

My overall takeaway is that I'm impressed by the model's performance in part two recognizing how people talk *about* assholes, considering how simple the methodology was (using just a bag-of-words approach and losing almost all semantic context). Perhaps unsurprisingly part one had much worse results, which could be attributed to how much less input there was (one post versus up-to-500 comments) and/or to how people tell stories when they suspect that they're actually the bad guy -- there's a lot of context/information left out until commenters ask for it or point inconsistencies out, and there's often a lot of indirect/obfuscating language and passive voice as well. It's probably telling that one of the words that occurred the most during the EDA of `asshole` posts was "technically".

My code and workflow are documented in the two 2B notebooks, where one is an overview of the best-performing models and the other a full rundown of all the models and sets of parameters I tried out.


### Part 2C: Bots

For better or worse, Reddit truly has subreddits for almost *everything*, including subreddits populated entirely by bots.

The first one I intended to work with is [r/SubredditSimulator](https://www.reddit.com/r/SubredditSimulator/), a "fully-automated subreddit that generates random submissions and comments using markov chains...with each bot account creating text based on comments from a different subreddit".

At the time of my updating this document, the top post is a link to a New York Times article about two sports teams no longer playing a certain artist's songs. The bot trained on [r/news](https://www.reddit.com/r/news), a subreddit for "real news articles, primarily but not exclusively, news relating to the United States and the rest of the World", shared that article with the post title **Exclusive: JPMorgan cuts ties to Muslim Brotherhood**. In the comments of the post, bots trained on various subreddits ranging from [r/hockey](https://www.reddit.com/r/hockey) to [r/dadjokes](https://www.reddit.com/r/dadjokes) to [r/programming](https://www.reddit.com/r/programming) gather to give their input.

My plan here was to collect posts from this subreddit from various bots and then collect *actual posts* written by humans from the original subreddits themselves, and see if I could make a model that could differentiate between them.