# Reddit Classification (trying to spot assholes and bots)

## This project is about pulling Reddit data and answering multiple classification questions based off of text data, using various supervised learning methods.

## Part 1 was data collection (gathering the Reddit data using the [Pushshift API](https://github.com/pushshift/api)) which can be found in [notebook #1](./1.%20Collecting%20Reddit%20Posts.ipynb). Part 2 is the fun stuff, including actual modeling.

### Part 2A: Love and Law

To start off, I decided to try to classify which subreddit (topic-based forum) a post came from, using two advice subreddits with fairly dissimilar content: [r/legaladvice](https://www.reddit.com/r/legaladvice) and [r/relationships](https://www.reddit.com/r/relationships). As the names suggest, one is a place “to ask simple legal questions, and to have legal concepts explained” and the other is a place for “helping people and…providing a platform for interpersonal relationship advice”. The general idea of each post is the original poster (OP) giving a description of their situation, and others respond in the comments with information and advice. The overall content is different enough that this isn’t a particularly difficult task. I pulled about 2,500 posts from each subreddit, cleaned the text up, and then grid-searched over a few different sets of parameters (the vectorizer used, the set of stop words, the size of the n-grams, number of features, the model itself).

Overall, the best model was a logistic regression, which performed better with an accuracy score of 97.3% whereas the multinomial naive Bayes had an accuracy score of 93.6%. The vectorizer that was used in the logistic regression model was CountVectorizer() with 500 features, filtering English stop words out, and looking at unigrams and bigrams.

My code and workflow are documented in [the 2A notebook](./2A.%20Classification%20-%20Love%20and%20Law.ipynb).

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

My code and workflow are documented in the two 2B notebooks, where [one is an overview of the best-performing models](./2B-Updated.%20Classification%20-%20Assholery%20(BEST).ipynb) and the other [a full rundown of all the models and sets of parameters I tried out](./2B-Updated.%20Classification%20-%20Assholery%20(FULL).ipynb).


### Part 2C: Bots

For better or worse, Reddit truly has subreddits for almost *everything*, including subreddits populated entirely by bots.

The first one I intended to work with is [r/SubredditSimulator](https://www.reddit.com/r/SubredditSimulator/), a "fully-automated subreddit that generates random submissions and comments using markov chains...with each bot account creating text based on comments from a different subreddit".

At the time of my updating this document, the top post is a link to a New York Times article about two sports teams no longer playing a certain artist's songs. The bot trained on [r/news](https://www.reddit.com/r/news), a subreddit for "real news articles, primarily but not exclusively, news relating to the United States and the rest of the World", shared that article with the post title **Exclusive: JPMorgan cuts ties to Muslim Brotherhood**. In the comments of the post, bots trained on various subreddits ranging from [r/hockey](https://www.reddit.com/r/hockey) to [r/dadjokes](https://www.reddit.com/r/dadjokes) to [r/programming](https://www.reddit.com/r/programming) gather to give their input.

My plan here was to collect bot posts and comments from this subreddit and then collect *actual posts and comments* written by humans from the original subreddits themselves, and then seee if I can train a model to differentiate between them. This is a more challenging problem than the previous two, because the content to be compared should more or less be the same in terms of general subject matter. Using a simple bag-of-words method wouldn't perform well because each bot is trained on subreddit-specific material and *seems* to be sharing relevant (if nonsensical to you or me) material. However, *because* each bot is trained only on its own subreddit, this means that any comment it makes is *also* about its subreddit topic. For example, here are five top-level comments on the same u/news_SS bot's post from before:

- u/hockey_SS on its hockey team (?), specific players, and a nonexistent hockey championship

- u/japan_SS on "sleeping on bare tatami with the lights always on"

- u/programming_SS on UI development across platforms

- u/fitness_SS on its ankle injury and frankly-impressive gainz ("500-750 is manageable")

- u/Aquariums_SS on the pros and cons of keeping goldfish versus shrimp

**This is the part that I'm working on currently -- I'm narrowing down the subreddits that I want to focus on for this portion.**

## Next Steps (in no particular order)

1. Replicating/building on part 2C but with [r/SubSimulatorGPT2](https://www.reddit.com/r/SubSimulatorGPT2/), a bot-populated subreddit where instead of using markov chains, "comments are generated automatically using a fine-tuned version of the GPT-2 language model developed by OpenAI". The content in this subreddit is *much* more coherent and realistic-seeming than in [r/SubredditSimulator](https://www.reddit.com/r/SubredditSimulator/), to the point that while looking into this sub I was legitimately a little horrified by some comments on an advice-seeking post until I remembered that they were all bot-generated.

2. Cleaning up the versions of the 2A and 2B notebooks where I do more than just simple bag-of-words (e.g., using word2vec instead of losing nearly all semantic information from context).
