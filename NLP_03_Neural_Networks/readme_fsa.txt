< -------------------- From article on Financial Sentiment Analysis (FSA)
FSA
    - define tasks and developing techniques (improve the performances using/curating human-annotated datasets)
    - use financial sentiment for applications on financial markets (discover appropriate applications)

Sentiment analysis
    - analyzes people’s sentiments, attitudes, opinions, emotions, evaluations, and appraisals
      towards various entities such as events, topics, services, products, individuals, organizations, issues, and their attributes.

    - started from article (2007, Yahoo! for Amazon: Sentiment extraction from small talk on the web)

Financial Sentiment Analysis (FSA) as a domain application for sentiment analysis
    - studies investor sentiment and financial textual sentiment
    - used to support business decision-making and perform financial forecasting
    - applications include corporate disclosures, annual reports, earning calls, financial news, social media interactions

Domain-dependence is more pronounced in the finance domain
    - because of topic concentration and the use of highly professional language.
    - For example, a word such as “liability” and “debt” is considered negative in general-purpose sentiment analysis, whereas it often has a neutral meaning in the financial contex.

Financial sentiment indicators
    - market-derived are computed proxies from market dynamics (e.g., price movement and trading volume)
      May include noise from other sources.
    - human-annotated sentiments, labeled by professionals or investors.

FSA research is shifting from human-annotated to market-derived sentiment
    - financial forecasting has become more popular in recent years.

Increase in interest <- increase in online materials such as digital news, World Wide Web, and social media.

FSA /  sentiment analysis, hypothesis testing and predictive modeling
Efficient Market Hypothesis (EMH) : financial markets are efficient and the price has incorporated all available market information.
The efficient market is classified into three forms:
    - strong (rare, but is useful for theoretical purposes)
      the price of securities reflects info including public, private, and historical price info
      with a presumption that it is free to trade and access info.

    - semi-strong
      stock prices reflect all public and historical information,
      while private information fails to influence market movements.

    - weak form
      the information set is merely historical prices, and
      any current or private information will not influence the market.

1. investors’ optimism or pessimism about future market activity:
   investor sentiment indicates the degree of deviation of an asset value from its economic fundamentals
   - American Association of Individual Investors (AAII) Investor Sentiment Survey
     insights into the perspectives of individual investors regarding the future direction of the market
     over a 6-month period through a weekly survey in which investors can vote Bullish, Neutral, or Bearish.

   - Sentix Investor Confidence
     the prospective economic outlook for the Eurozone over a 6-month period
     is derived from a comprehensive survey involving investors and analysts
        - a reading surpassing zero - signifies a positive / optimistic outlook
        - a reading below - indicates a negative / pessimistic perspective.

   - Investors Intelligence Sentiment Index
     operates on contrarian principles
     conducts surveys of 100+ independent market newsletters
     evaluate the current stance of each author regarding the market, whether it be bullish, bearish, or indicating a correction.

Market Sentiment (also investor sentiment)
    - general outlook or attitude of investors toward a particular security or the overall financial market.
      The optimism or pessimism of the market players is most evident in the overall price trends.

   Investor sentiment can also be measured through textual data such as microblogs and analyst reports
   also derived from
     - social networks (X / formerly known as Twitter, StockTwits or Facebook)
     - message boards (RagingBull.com, Yahoo!, Finance) or
     - Google searches.

2. Financial (textual) sentiment is measured by the degree of positivity or negativity in financial texts.
   - subjective judgment and analysis from investors and analysts (lagging)
     published on social media and self-media.

   - objective info (leading / influential to investors’ judgment):
     political and macroeconomics news, break news, and annual reports released by companies
     objective reflection of conditions within the general environment, industries, markets, and firms

Financial textual sentiment analysis <> classic sentiment analysis:
    a. it involves the frequent use of metaphorical expressions in financial communication
       Ex. “The market is riding a bull” is a common metaphor signifying a robust, upward market movement.

    b. Precision and brevity, concise language to convey complex info
       so the need to decode sentiments from concise sentence structures
       Ex. Not “The company experienced a substantial increase in revenue and a corresponding improvement in profitability,”
       but rather “The company posted robust revenue growth, driving higher profits.”

    c. The financial industry employs a unique set of terms and jargon with specific meanings.
       so need to understand for accurate interpretation and analysis of financial texts
       Ex. the “Price-to-Earnings (P/E) ratio" is a fundamental financial metric used to assess a company’s valuation.
       A high P/E ratio may indicate that investors hold high expectations for future earnings.

    d. Unlike classic sentiment analysis (which typically focuses on text alone)
        financial texts often integrate qualitative text with quantitative data,
        - to understand the language used in financial texts but also
        - to process and analyze numerical information in conjunction with the textual context,
    to gain a comprehensive understanding of the sentiment.

    e. Direction-dependent and the direction of events or changes
       Ex. the word “profit” may carry both positive and negative sentiment depending on the direction.
        - an increase in profit is generally regarded as positive,
        - a decrease is seen as negative.

In practice, there are six areas that cause FSA fail (provide context and nuance)
    - irrealis mood (conditional mood, subjunctive mood, imperative mood),
    - rhetoric (negative assertion, personification, sarcasm),
    - dependent opinion,
    - unspecified aspects,
    - unrecognized words (entity, microtext, jargons), and
    - external reference.

3. Market sentiment
- often used interchangeably with investor sentiment
- but investors may hold varying viewpoints at different periods and markets
- is the collective outlook of investors towards a specific financial market or security.

- Ascending prices signal an optimistic or bullish market sentiment,
- Descending prices signal a pessimistic or bearish market sentiment.

Market sentiment can be measured by proxy financial metrics
 (backward-looking, lagging indicators) the degree of price movements and volatility computed from historical market data, and thus
    - Chicago Board Options Exchange (CBOE) Volatility Index (VIX)
      - measures expected market volatility based on real-time prices of the S&P 500 Index options over the next 30 days.
      - is higher when there is a greater level of fear and uncertainty in the market
      - lower in bull markets.
      - Options on S&P 500 futures are contracts that give the buyer the right, but not the obligation,
        to buy (for a call option) or sell (for a put option) the underlying S&P 500 futures contract
        at a specified strike price and expiration date.

   - Baker and Wurgler Index
     - is generated from the first principal component of six proxies from market variables which are
       CEFD, dividend premium, equity issues, first-day return, IPO activity, and trading volume.

   - Equity Market Sentiment Index (EMSI)
   - High-Low Index, Bullish Percent Index (BPI)

4. FSA Research Scope studies (interconnected)
    - technique-driven : sentiment analysis of financial texts
    - (unique to finance) application-driven :
        proxy of investor sentiment to make predictions in financial markets.

The benchmark datasets for FSA technique studies
    - usually require human annotation with sentiment polarities or intensity scores,
    - which need to be sufficient, representative, and precise to train an unbiased model in different domains.

Data sources for FSA application studies are
    - annotated by financial metrics, computed from the market data
    - require the data to be in a time series with
        - financial texts and
        - representative periods
      to model the relationships between investor sentiment and financial metrics.

FSA techniques focus on level of sentiment at which the sentiment is detected
    - targeted aspect-based sentiment analysis vs. sentence-level sentiment analysis.

FSA applications explore financial application scenarios, such as
    - stock market movement prediction,
    - financial risk prediction,
    - portfolio management : require more complex methods (e.g., reinforcement learning) and evaluations (e.g., trading simulation)
    - FOREX market prediction, and
    - cryptocurrency market prediction.

Investor sentiment can be measured by
    - financial textual sentiment,
    - sentiment surveys, and
    - indices constructed from market data.

FSA
    - analyzes people’s sentiment from financial texts
    - measures and quantifies investor sentiment from financial textual sentiment
    - is grounded in the applications of market prediction and financial decision-making.

Categories of research in FSA
    - study the techniques to improve the performance of tasks as
        - paragraph and sentence-level sentiment analysis
        - (targeted) aspect-based sentiment analysis and
        - development of financial lexicons and sentiment analysis models
    - application-driven or market-driven
        - causality and correlation testing and financial forecasting.

The sentiment
    - represented in an explicit : generation of sentiment words, polarity, or intensity score
    - implicit : generation of feature embedding.

Sentiment analysis can be performed in
    - coarse-grained manner : granularity and expression.
    - fine-grained

Granularity : level of sentiment at which the sentiment is detected and it includes
    - document-level
    - paragraph-level
    - sentence-level
    - aspect-level (Aspect-based FSA)

The task is to detect the text that is favorable or unfavorable to a specific given target.

Targeted Aspect-based FSA (TABFSA) : the most challenging but pragmatic task is called
    - aims to extract entities and aspects and detect their corresponding sentiment in financial texts.

Sentiment analysis
    - polarity detection fashion (classification to positive or negative)
    - intensity score

Benchmark Datasets
    - textual data :
        - email communications
        - social media posts (e.g., tweets)
        - corporate reports, and
        - daily news
Financial corpora
    - are labeled through manual annotation
    - based on stock price

There is one document-level, four sentence-level, two target-level, and one targeted aspect-level dataset.
The annotation is becoming more granular on target- and aspect-level.

Open challenges such as SemEval 2017 Task 5 and FiQA Task 1.

The annotation is a challenging task as it is subject to human factors
    - domain expertise
    - the annotator’s private state
    - inference made into the text

Benchmark Dataset   | Number of Entries                             | Type of Annotation            | Data Source
--------------------------------------------------------------------------------------------------------------------
PhraseBank	        | Entries: 4,846                                | Sentence-level polarity       | News headline
                    | 100% Agree (pos: 570, neu: 1391, neg: 303)    | (positive, neutral, negative)
                    | 75% Agree (pos: 887, neu: 2146, neg: 420)
                    | 66% Agree (pos: 1,168, neu: 2,535, neg: 514)
                    | 50% Agree (pos: 1,363, neu: 2,879, neg: 604)
--------------------------------------------------------------------------------------------------------------------
SemEval 2017        | Entries: 2,836                                | Targeted sentence-level   | News headline
Task 5	            | Headline (pos: 653, neu: 38, neg: 451)        | sentiment score	        | and microblogs
                    | Microblogs (pos: 1,086, neu: 27, neg: 581)
--------------------------------------------------------------------------------------------------------------------
FiQA Task 1	        | Entries: 1,173                                | Targeted aspect-level     | News headline
                    | Headline (pos: 320, neu: 13, neg:165)         | sentiment score	        | and posts
                    | Post (pos: 440, neu: 1, neg:234)
--------------------------------------------------------------------------------------------------------------------
Topic-Specific      | Entries: 297              | Targeted document-level           | News
Sentiment Analysis  |                           | sentiment polarity                |
                    |                           | (very negative, negative,         |
                    |                           | slightly negative, neutral,       |
                    |                           | slightly positive, positive,      |
                    |                           | very positive)                    |
--------------------------------------------------------------------------------------------------------------------
StockSen	        | Entries: 55,171	        | Sentence-level polarity           | StockTwits
                    |                           | (positive, neutral, negative)     |
--------------------------------------------------------------------------------------------------------------------
SentiEcon GS-1000	| Entries: 1,000	        | Sentence-level polarity           | Business daily news
                    |                           | (positive, negative, none)        |
--------------------------------------------------------------------------------------------------------------------
FinLin	            | Entries: 3,811	        | Sentence-level sentiment score    | Stocktwits, news articles, company
                    |                           |                                   | reports and investor reports
--------------------------------------------------------------------------------------------------------------------
SEntFiN	            | Entries: 10,753                       | Targeted sentence-level polarity  | News headline
                    | pos: 5,074, neu: 5,517, neg: 3,814    | (positive, neutral, negative)
--------------------------------------------------------------------------------------------------------------------


-------------- END of article>


Kelvin Du, Frank Xing, Rui Mao, and Erik Cambria. 2024.
Financial Sentiment Analysis: Techniques and Applications. ACM Comput. Surv. 56, 9, Article 220 (September 2024), 42 pages.
https://doi.org/10.1145/3649451

https://www.kaggle.com/ankurzing/sentiment-analysis-for-financial-news

https://www.idrbt.ac.in/wp-content/uploads/2022/07/AI_2020.pdf

https://github.com/huggingface/datasets/blob/main/docs/README.md
https://www.tensorflow.org/datasets/community_catalog/huggingface/financial_phrasebank#sentences_allagree

https://towardsdatascience.com/an-easy-tutorial-about-sentiment-analysis-with-deep-learning-and-keras-2bf52b9cba91/
