# ML for Trading - 2<sup>nd</sup> Edition

这本书（https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715?pf_rd_r=GZH2XZ35GB3BET09PCCA&pf_rd_p=c5b6893a-24f2-4a59-9d4b-aff5065c90ec&pd_rd_r=91 a679c7 - f069-4a6e- bdbb -a2b3f548f0c8&pd_rd_w=2B0Q0&pd_rd_wg=GMY5S&ref_=pd_gw_ci_mcx_mr_hp_d ）旨在展示机器学习如何以实用而全面的方式为算法交易策略增加价值。它涵盖了从线性回归到深度强化学习的广泛机器学习技术，并演示了如何构建、回测和评估由模型预测驱动的交易策略。  

分为四部分，** 23 章加上附录**，涵盖** 800 多页**：
-数据源、**金融特征工程**和投资组合管理的重要方面，
-基于监督和无监督机器学习算法**的多空**策略的设计和评估，
-- 如何从**财务文本数据**（如 SEC 文件、财报电话会议记录或财经新闻）中提取可交易信号，**financial text data** like SEC filings, earnings call transcripts or financial news,
- using **deep learning** models like CNN and RNN with market and alternative data, how to generate synthetic data with generative adversarial networks, and training a trading agent using deep reinforcement learning

<p align="center">
<a href="https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715?pf_rd_r=GZH2XZ35GB3BET09PCCA&pf_rd_p=c5b6893a-24f2-4a59-9d4b-aff5065c90ec&pd_rd_r=91a679c7-f069-4a6e-bdbb-a2b3f548f0c8&pd_rd_w=2B0Q0&pd_rd_wg=GMY5S&ref_=pd_gw_ci_mcx_mr_hp_d">
<img src="https://ml4t.s3.amazonaws.com/assets/cover_toc_gh.png" width="75%">
</a>
</p>

This repo contains **over 150 notebooks** that put the concepts, algorithms, and use cases discussed in the book into action. They provide numerous examples that show:
-- 如何使用市场、基本面和替代文本和图像数据并从中提取信号，
-- 如何训练和调整模型来预测不同资产类别和投资期限的回报，包括如何复制最近发表的研究，以及
-- 如何设计、回测和评估交易策略。

>> 我们**强烈建议**在阅读本书时复习笔记本；它们通常处于已执行状态，并且通常包含由于空间限制而未包含的附加信息。  **highly recommend** reviewing the notebooks while reading the book; they are usually in an executed state and often contain additional information not included due to space constraints.  

除了本存储库中的信息之外，本书的[网站](ml4trading.io) 还包含章节摘要和其他信息。[website](ml4trading.io) contains chapter summary and additional information.

## Join the ML4T Community!

为了方便读者就本书的内容和代码示例以及自己的策略和行业发展的制定和实施提出问题，我们正在托管一个在线[平台](https://exchange.ml4trading.io /）。[platform](https://exchange.ml4trading.io/).

请[加入](https://exchange.ml4trading.io/)我们的社区，与有兴趣利用机器学习进行交易策略的其他交易者联系，分享您的经验，并互相学习！[join](https://exchange.ml4trading.io/) our community and connect with fellow traders interested in leveraging ML for trading strategies, share your experience, and learn from each other! 

## What's new in the 2<sup>nd</sup> Edition?

First and foremost, this [book](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715?pf_rd_r=VMKJPZC4N36TTZZCWATP&pf_rd_p=c5b6893a-24f2-4a59-9d4b-aff5065c90ec&pd_rd_r=8f331266-0d21-4c76-a3eb-d2e61d23bb31&pd_rd_w=kVGNF&pd_rd_wg=LYLKH&ref_=pd_gw_ci_mcx_mr_hp_d) demonstrates how you can extract signals from a diverse set of data sources and design trading strategies for different asset classes using a broad range of supervised, unsupervised, and reinforcement learning algorithms。它还提供相关的数学和统计知识，以方便算法的调整或结果的解释。此外，它还涵盖了金融背景，可帮助您处理市场和基本数据、提取信息特征以及管理交易策略的绩效。

从实践的角度来看，第二版旨在为您提供概念理解和工具，以开发您自己的基于机器学习的交易策略。为此，它将机器学习视为流程中的关键元素，而不是独立的练习，引入了从数据源、特征工程、模型优化到策略设计和回测的交易工作流程的端到端机器学习。

More specifically, the ML4T workflow starts with generating ideas for a well-defined investment universe, collecting relevant data, and extracting informative features. It also involves designing, tuning, and evaluating ML models suited to the predictive task. Finally, it requires developing trading strategies to act on the models' predictive signals, as well as simulating and evaluating their performance on historical data using a backtesting engine. Once you decide to execute an algorithmic strategy in a real market, you will find yourself iterating over this workflow repeatedly to incorporate new information and a changing environment.

<p align="center">
<img src="https://i.imgur.com/kcgItgp.png" width="75%">
</p>

The [second edition](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715?pf_rd_r=GZH2XZ35GB3BET09PCCA&pf_rd_p=c5b6893a-24f2-4a59-9d4b-aff5065c90ec&pd_rd_r=91a679c7-f069-4a6e-bdbb-a2b3f548f0c8&pd_rd_w=2B0Q0&pd_rd_wg=GMY5S&ref_=pd_gw_ci_mcx_mr_hp_d)'s emphasis on the ML4t workflow translates into a new chapter on [strategy backtesting](08_ml4t_workflow), a new [appendix](24_alpha_factor_library) describing over 100 different alpha factors, and many new practical applications. We have also rewritten most of the existing content for clarity and readability. 

The trading applications now use a broader range of data sources beyond daily US equity prices, including international stocks and ETFs. It also demonstrates how to use ML for an intraday strategy with minute-frequency equity data. Furthermore, it extends the coverage of alternative data sources to include SEC filings for sentiment analysis and return forecasts, as well as satellite images to classify land use. 

Another innovation of the second edition is to replicate several trading applications recently published in top journals: 
- [Chapter 18](18_convolutional_neural_nets) demonstrates how to apply convolutional neural networks to time series converted to image format for return predictions based on [Sezer and Ozbahoglu](https://www.researchgate.net/publication/324802031_Algorithmic_Financial_Trading_with_Deep_Convolutional_Neural_Networks_Time_Series_to_Image_Conversion_Approach) (2018). 
- [Chapter 20](20_autoencoders_for_conditional_risk_factors) shows how to extract risk factors conditioned on stock characteristics for asset pricing using autoencoders based on [Autoencoder Asset Pricing Models](https://www.aqr.com/Insights/Research/Working-Paper/Autoencoder-Asset-Pricing-Models) by Shihao Gu, Bryan T. Kelly, and Dacheng Xiu (2019), and 
-- [第21章](21_gans_for_synthetic_time_series)展示了如何使用基于[时间序列生成对抗网络]的生成对抗网络创建合成训练数据(https://papers.nips.cc/paper/8789-time-series-generative-对抗网络），作者：Jinsung Yoon、Daniel Jarrett 和 Mihaela van der Schaar（2019）。[Chapter 21](21_gans_for_synthetic_time_series) shows how to create synthetic training data using generative adversarial networks based on [Time-series Generative Adversarial Networks](https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks) by Jinsung Yoon, Daniel Jarrett, and Mihaela van der Schaar (2019).

All applications now use the latest available (at the time of writing) software versions such as pandas 1.0 and TensorFlow 2.2. There is also a customized version of Zipline that makes it easy to include machine learning model predictions when designing a trading strategy.

## Installation, data sources and bug reports

The code examples rely on a wide range of Python libraries from the data science and finance domains.

It is not necessary to try and install all libraries at once because this increases the likeliihood of encountering version conflicts. Instead, we recommend that you install the libraries required for a specific chapter as you go along.

> Update March 2022: `zipline-reloaded`, `pyfolio-reloaded`, `alphalens-reloaded`, and `empyrical-reloaded` are now available on the `conda-forge` channel. The channel `ml4t` only contains outdated versions and will soon be removed.

> Update April 2021: with the update of [Zipline](https://zipline.ml4trading.io), it is no longer necessary to use Docker. The installation instructions now refer to OS-specific environment files that should simplify your running of the notebooks.

> Update Februar 2021: code sample release 2.0 updates the conda environments provided by the Docker image to Python 3.8, Pandas 1.2, and TensorFlow 1.2, among others; the Zipline backtesting environment with now uses Python 3.6.

- The [installation](installation/README.md) directory contains detailed instructions on setting up and using a Docker image to run the notebooks. It also contains configuration files for setting up various `conda` environments and install the packages used in the notebooks directly on your machine if you prefer (and, depending on your system, are prepared to go the extra mile).
- To download and preprocess many of the data sources used in this book, see the instructions in the [README](data/README.md) file alongside various notebooks in the [data](data) directory.

> If you have any difficulties installing the environments, downloading the data or running the code, please raise a **GitHub issue** in the repo ([here](https://github.com/stefan-jansen/machine-learning-for-trading/issues)). Working with GitHub issues has been described [here](https://guides.github.com/features/issues/).

> **Update**: You can download the **[algoseek](https://www.algoseek.com)** data used in the book [here](https://www.algoseek.com/ml4t-book-data.html). See instructions for preprocessing in [Chapter 2](02_market_and_fundamental_data/02_algoseek_intraday/README.md) and an intraday example with a gradient boosting model in [Chapter 12](12_gradient_boosting_machines/10_intraday_features.ipynb).  

> **Update**: The [figures](figures) directory contains color versions of the charts used in the book. 

# Outline & Chapter Summary

The [book](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715?pf_rd_r=GZH2XZ35GB3BET09PCCA&pf_rd_p=c5b6893a-24f2-4a59-9d4b-aff5065c90ec&pd_rd_r=91a679c7-f069-4a6e-bdbb-a2b3f548f0c8&pd_rd_w=2B0Q0&pd_rd_wg=GMY5S&ref_=pd_gw_ci_mcx_mr_hp_d) has four parts that address different challenges that arise when sourcing and working with market, fundamental and alternative data sourcing, developing ML solutions to various predictive tasks in the trading context, and designing and evaluating a trading strategy that relies on predictive signals generated by an ML model.

> The directory for each chapter contains a README with additional information on content, code examples and additional resources.  

[Part 1: From Data to Strategy Development](#part-1-from-data-to-strategy-development)
* [01 Machine Learning for Trading: From Idea to Execution](#01-machine-learning-for-trading-from-idea-to-execution)
* [02 Market & Fundamental Data: Sources and Techniques](#02-market--fundamental-data-sources-and-techniques)
* [03 Alternative Data for Finance: Categories and Use Cases](#03-alternative-data-for-finance-categories-and-use-cases)
* [04 Financial Feature Engineering: How to research Alpha Factors](#04-financial-feature-engineering-how-to-research-alpha-factors)
* [05 Portfolio Optimization and Performance Evaluation](#05-portfolio-optimization-and-performance-evaluation)

[Part 2: Machine Learning for Trading: Fundamentals](#part-2-machine-learning-for-trading-fundamentals)
* [06 The Machine Learning Process](#06-the-machine-learning-process)
* [07 Linear Models: From Risk Factors to Return Forecasts](#07-linear-models-from-risk-factors-to-return-forecasts)
* [08 The ML4T Workflow: From Model to Strategy Backtesting](#08-the-ml4t-workflow-from-model-to-strategy-backtesting)
* [09 Time Series Models for Volatility Forecasts and Statistical Arbitrage](#09-time-series-models-for-volatility-forecasts-and-statistical-arbitrage)
* [10 Bayesian ML: Dynamic Sharpe Ratios and Pairs Trading](#10-bayesian-ml-dynamic-sharpe-ratios-and-pairs-trading)
* [11 Random Forests: A Long-Short Strategy for Japanese Stocks](#11-random-forests-a-long-short-strategy-for-japanese-stocks)
* [12 Boosting your Trading Strategy](#12-boosting-your-trading-strategy)
* [13 Data-Driven Risk Factors and Asset Allocation with Unsupervised Learning](#13-data-driven-risk-factors-and-asset-allocation-with-unsupervised-learning)

[Part 3: Natural Language Processing for Trading](#part-3-natural-language-processing-for-trading)
* [14 Text Data for Trading: Sentiment Analysis](#14-text-data-for-trading-sentiment-analysis)
* [15 Topic Modeling: Summarizing Financial News](#15-topic-modeling-summarizing-financial-news)
* [16 Word embeddings for Earnings Calls and SEC Filings](#16-word-embeddings-for-earnings-calls-and-sec-filings)

[Part 4: Deep & Reinforcement Learning](#part-4-deep--reinforcement-learning)
* [17 Deep Learning for Trading](#17-deep-learning-for-trading)
* [18 CNN for Financial Time Series and Satellite Images](#18-cnn-for-financial-time-series-and-satellite-images)
* [19 RNN for Multivariate Time Series and Sentiment Analysis](#19-rnn-for-multivariate-time-series-and-sentiment-analysis)
* [20 Autoencoders for Conditional Risk Factors and Asset Pricing](#20-autoencoders-for-conditional-risk-factors-and-asset-pricing)
* [21 Generative Adversarial Nets for Synthetic Time Series Data](#21-generative-adversarial-nets-for-synthetic-time-series-data)
* [22 Deep Reinforcement Learning: Building a Trading Agent](#22-deep-reinforcement-learning-building-a-trading-agent)
* [23 Conclusions and Next Steps](#23-conclusions-and-next-steps)
* [24 Appendix - Alpha Factor Library](#24-appendix---alpha-factor-library)

## Part 1: From Data to Strategy Development

The first part provides a framework for developing trading strategies driven by machine learning (ML). It focuses on the data that power the ML algorithms and strategies discussed in this book, outlines how to engineer and evaluates features suitable for ML models, and how to manage and measure a portfolio's performance while executing a trading strategy.

### 01 Machine Learning for Trading: From Idea to Execution

This [chapter](01_machine_learning_for_trading) explores industry trends that have led to the emergence of ML as a source of competitive advantage in the investment industry. We will also look at where ML fits into the investment process to enable algorithmic trading strategies. 

More specifically, it covers the following topics:
- Key trends behind the rise of ML in the investment industry
- The design and execution of a trading strategy that leverages ML
- Popular use cases for ML in trading

### 02 Market & Fundamental Data: Sources and Techniques

This [chapter](02_market_and_fundamental_data) shows how to work with market and fundamental data and describes critical aspects of the environment that they reflect. For example, familiarity with various order types and the trading infrastructure matter not only for the interpretation of the data but also to correctly design backtest simulations. We also illustrate how to use Python to access and manipulate trading and financial statement data.  

Practical examples demonstrate how to work with trading data from NASDAQ tick data and Algoseek minute bar data with a rich set of attributes capturing the demand-supply dynamic that we will later use for an ML-based intraday strategy. We also cover various data provider APIs and how to source financial statement information from the SEC.

<p align="center">
<img src="https://i.imgur.com/enaSo0C.png" title="Order Book" width="50%"/>
</p>
In particular, this chapter covers:

- How market data reflects the structure of the trading environment
- Working with intraday trade and quotes data at minute frequency
- Reconstructing the **limit order book** from tick data using NASDAQ ITCH 
- Summarizing tick data using various types of bars
- Working with eXtensible Business Reporting Language (XBRL)-encoded **electronic filings**
- Parsing and combining market and fundamental data to create a P/E series
- How to access various market and fundamental data sources using Python

### 03 Alternative Data for Finance: Categories and Use Cases

This [chapter](03_alternative_data) outlines categories and use cases of alternative data, describes criteria to assess the exploding number of sources and providers, and summarizes the current market landscape. 

It also demonstrates how to create alternative data sets by scraping websites, such as collecting earnings call transcripts for use with natural language processing (NLP) and sentiment analysis algorithms in the third part of the book.
 
More specifically, this chapter covers:

- Which new sources of signals have emerged during the alternative data revolution
- How individuals, business, and sensors generate a diverse set of alternative data
- Important categories and providers of alternative data
- Evaluating how the burgeoning supply of alternative data can be used for trading
- Working with alternative data in Python, such as by scraping the internet

### 04 Financial Feature Engineering: How to research Alpha Factors

If you are already familiar with ML, you know that feature engineering is a crucial ingredient for successful predictions. It matters at least as much in the trading domain, where academic and industry researchers have investigated for decades what drives asset markets and prices, and which features help to explain or predict price movements.

<p align="center">
<img src="https://i.imgur.com/UCu4Huo.png" width="70%">
</p>

This [chapter](04_alpha_factor_research) outlines the key takeaways of this research as a starting point for your own quest for alpha factors. It also presents essential tools to compute and test alpha factors, highlighting how the NumPy, pandas, and TA-Lib libraries facilitate the manipulation of data and present popular smoothing techniques like the wavelets and the Kalman filter that help reduce noise in data. After reading it, you will know about:
- Which categories of factors exist, why they work, and how to measure them,
- Creating alpha factors using NumPy, pandas, and TA-Lib,
- How to de-noise data using wavelets and the Kalman filter,
- Using Zipline to test individual and multiple alpha factors,
- How to use [Alphalens](https://github.com/quantopian/alphalens) to evaluate predictive performance.
 
### 05 Portfolio Optimization and Performance Evaluation

Alpha factors generate signals that an algorithmic strategy translates into trades, which, in turn, produce long and short positions. The returns and risk of the resulting portfolio determine whether the strategy meets the investment objectives.
<p align="center">
<img src="https://i.imgur.com/E2h63ZB.png" width="65%">
</p>

There are several approaches to optimize portfolios. These include the application of machine learning (ML) to learn hierarchical relationships among assets and treat them as complements or substitutes when designing the portfolio's risk profile. This [chapter](05_strategy_evaluation) covers:
- How to measure portfolio risk and return
- Managing portfolio weights using mean-variance optimization and alternatives
- Using machine learning to optimize asset allocation in a portfolio context
- Simulating trades and create a portfolio based on alpha factors using Zipline
- How to evaluate portfolio performance using [pyfolio](https://quantopian.github.io/pyfolio/)

## Part 2: Machine Learning for Trading: Fundamentals

The second part covers the fundamental supervised and unsupervised learning algorithms and illustrates their application to trading strategies. It also introduces the Quantopian platform that allows you to leverage and combine the data and ML techniques developed in this book to implement algorithmic strategies that execute trades in live markets.

### 06 The Machine Learning Process

This [chapter](06_machine_learning_process) kicks off Part 2 that illustrates how you can use a range of supervised and unsupervised ML models for trading. We will explain each model's assumptions and use cases before we demonstrate relevant applications using various Python libraries. 

There are several aspects that many of these models and their applications have in common. This chapter covers these common aspects so that we can focus on model-specific usage in the following chapters. It sets the stage by outlining how to formulate, train, tune, and evaluate the predictive performance of ML models as a systematic workflow. The content includes:

<p align="center">
<img src="https://i.imgur.com/5qisClE.png" width="65%">
</p>

- How supervised and unsupervised learning from data works
- Training and evaluating supervised learning models for regression and classification tasks
- How the bias-variance trade-off impacts predictive performance
- How to diagnose and address prediction errors due to overfitting
- Using cross-validation to optimize hyperparameters with a focus on time-series data
- Why financial data requires additional attention when testing out-of-sample

### 07 Linear Models: From Risk Factors to Return Forecasts

Linear models are standard tools for inference and prediction in regression and classification contexts. Numerous widely used asset pricing models rely on linear regression. Regularized models like Ridge and Lasso regression often yield better predictions by limiting the risk of overfitting. Typical regression applications identify risk factors that drive asset returns to manage risks or predict returns. Classification problems, on the other hand, include directional price forecasts.

<p align="center">
<img src="https://i.imgur.com/3Ph6jma.png" width="65%">
</p>

[Chapter 07](07_linear_models) covers the following topics:

- How linear regression works and which assumptions it makes
- Training and diagnosing linear regression models
- Using linear regression to predict stock returns
- Use regularization to improve the predictive performance
- How logistic regression works
- Converting a regression into a classification problem

### 08 The ML4T Workflow: From Model to Strategy Backtesting

This [chapter](08_ml4t_workflow) presents an end-to-end perspective on designing, simulating, and evaluating a trading strategy driven by an ML algorithm. 
We will demonstrate in detail how to backtest an ML-driven strategy in a historical market context using the Python libraries [backtrader](https://www.backtrader.com/) and [Zipline](https://zipline.ml4trading.io/index.html). 
The ML4T workflow ultimately aims to gather evidence from historical data that helps decide whether to deploy a candidate strategy in a live market and put financial resources at risk. A realistic simulation of your strategy needs to faithfully represent how security markets operate and how trades execute. Also, several methodological aspects require attention to avoid biased results and false discoveries that will lead to poor investment decisions.

<p align="center">
<img src="https://i.imgur.com/R9O0fn3.png" width="65%">
</p>

More specifically, after working through this chapter you will be able to:

- Plan and implement end-to-end strategy backtesting
- Understand and avoid critical pitfalls when implementing backtests
- Discuss the advantages and disadvantages of vectorized vs event-driven backtesting engines
- Identify and evaluate the key components of an event-driven backtester
- Design and execute the ML4T workflow using data sources at minute and daily frequencies, with ML models trained separately or as part of the backtest
- Use Zipline and backtrader to design and evaluate your own strategies 

### 09 Time Series Models for Volatility Forecasts and Statistical Arbitrage

This [chapter](09_time_series_models) focuses on models that extract signals from a time series' history to predict future values for the same time series. 
Time series models are in widespread use due to the time dimension inherent to trading. It presents tools to diagnose time series characteristics such as stationarity and extract features that capture potentially useful patterns. It also introduces univariate and multivariate time series models to forecast macro data and volatility patterns. 
Finally, it explains how cointegration identifies common trends across time series and shows how to develop a pairs trading strategy based on this crucial concept. 

<p align="center">
<img src="https://i.imgur.com/cglLgJ0.png" width="90%">
</p>

In particular, it covers:
- How to use time-series analysis to prepare and inform the modeling process
- Estimating and diagnosing univariate autoregressive and moving-average models
- Building autoregressive conditional heteroskedasticity (ARCH) models to predict volatility
- How to build multivariate vector autoregressive models
- Using cointegration to develop a pairs trading strategy

### 10 Bayesian ML: Dynamic Sharpe Ratios and Pairs Trading

Bayesian statistics allows us to quantify uncertainty about future events and refine estimates in a principled way as new information arrives. This dynamic approach adapts well to the evolving nature of financial markets. 
Bayesian approaches to ML enable new insights into the uncertainty around statistical metrics, parameter estimates, and predictions. The applications range from more granular risk management to dynamic updates of predictive models that incorporate changes in the market environment. 

<p align="center">
<img src="https://i.imgur.com/qOUPIDV.png" width="80%">
</p>

More specifically, this [chapter](10_bayesian_machine_learning) covers: 
- How Bayesian statistics applies to machine learning
- Probabilistic programming with PyMC3
- Defining and training machine learning models using PyMC3
- How to run state-of-the-art sampling methods to conduct approximate inference
- Bayesian ML applications to compute dynamic Sharpe ratios, dynamic pairs trading hedge ratios, and estimate stochastic volatility


### 11 Random Forests: A Long-Short Strategy for Japanese Stocks

This [chapter](11_decision_trees_random_forests) applies decision trees and random forests to trading. Decision trees learn rules from data that encode nonlinear input-output relationships. We show how to train a decision tree to make predictions for regression and classification problems, visualize and interpret the rules learned by the model, and tune the model's hyperparameters to optimize the bias-variance tradeoff and prevent overfitting.

The second part of the chapter introduces ensemble models that combine multiple decision trees in a randomized fashion to produce a single prediction with a lower error. It concludes with a long-short strategy for Japanese equities based on trading signals generated by a random forest model.

<p align="center">
<img src="https://i.imgur.com/S4s0rou.png" width="80%">
</p>

In short, this chapter covers:
- Use decision trees for regression and classification
- Gain insights from decision trees and visualize the rules learned from the data
- Understand why ensemble models tend to deliver superior results
- Use bootstrap aggregation to address the overfitting challenges of decision trees
- Train, tune, and interpret random forests
- Employ a random forest to design and evaluate a profitable trading strategy


### 12 Boosting your Trading Strategy

Gradient boosting is an alternative tree-based ensemble algorithm that often produces better results than random forests. The critical difference is that boosting modifies the data used to train each tree based on the cumulative errors made by the model. While random forests train many trees independently using random subsets of the data, boosting proceeds sequentially and reweights the data.
This [chapter](12_gradient_boosting_machines) shows how state-of-the-art libraries achieve impressive performance and apply boosting to both daily and high-frequency data to backtest an intraday trading strategy. 

<p align="center">
<img src="https://i.imgur.com/Re0uI0H.png" width="70%">
</p>

More specifically, we will cover the following topics:
- How does boosting differ from bagging, and how did gradient boosting evolve from adaptive boosting,
- Design and tune adaptive and gradient boosting models with scikit-learn,
- Build, optimize, and evaluate gradient boosting models on large datasets with the state-of-the-art implementations XGBoost, LightGBM, and CatBoost,
- Interpreting and gaining insights from gradient boosting models using [SHAP](https://github.com/slundberg/shap) values, and
- Using boosting with high-frequency data to design an intraday strategy.

### 13 Data-Driven Risk Factors and Asset Allocation with Unsupervised Learning

Dimensionality reduction and clustering are the main tasks for unsupervised learning: 
- Dimensionality reduction transforms the existing features into a new, smaller set while minimizing the loss of information. A broad range of algorithms exists that differ by how they measure the loss of information, whether they apply linear or non-linear transformations or the constraints they impose on the new feature set. 
- Clustering algorithms identify and group similar observations or features instead of identifying new features. Algorithms differ in how they define the similarity of observations and their assumptions about the resulting groups.

<p align="center">
<img src="https://i.imgur.com/Rfk7uCM.png" width="70%">
</p>

More specifically, this [chapter](13_unsupervised_learning) covers:
- How principal and independent component analysis (PCA and ICA) perform linear dimensionality reduction
- Identifying data-driven risk factors and eigenportfolios from asset returns using PCA
- Effectively visualizing nonlinear, high-dimensional data using manifold learning
- Using T-SNE and UMAP to explore high-dimensional image data
- How k-means, hierarchical, and density-based clustering algorithms work
- Using agglomerative clustering to build robust portfolios with hierarchical risk parity


## Part 3: Natural Language Processing for Trading

Text data are rich in content, yet unstructured in format and hence require more preprocessing so that a machine learning algorithm can extract the potential signal. The critical challenge consists of converting text into a numerical format for use by an algorithm, while simultaneously expressing the semantics or meaning of the content. 

The next three chapters cover several techniques that capture language nuances readily understandable to humans so that machine learning algorithms can also interpret them.

### 14 Text Data for Trading: Sentiment Analysis

Text data is very rich in content but highly unstructured so that it requires more preprocessing to enable an ML algorithm to extract relevant information. A key challenge consists of converting text into a numerical format without losing its meaning.
This [chapter](14_working_with_text_data) shows how to represent documents as vectors of token counts by creating a document-term matrix that, in turn, serves as input for text classification and sentiment analysis. It also introduces the Naive Bayes algorithm and compares its performance to linear and tree-based models.

In particular, in this chapter covers:
- What the fundamental NLP workflow looks like
- How to build a multilingual feature extraction pipeline using spaCy and TextBlob
- Performing NLP tasks like part-of-speech tagging or named entity recognition
- Converting tokens to numbers using the document-term matrix
- Classifying news using the naive Bayes model
- How to perform sentiment analysis using different ML algorithms

### 15 Topic Modeling: Summarizing Financial News

This [chapter](15_topic_modeling) uses unsupervised learning to model latent topics and extract hidden themes from documents. These themes can generate detailed insights into a large corpus of financial reports.
Topic models automate the creation of sophisticated, interpretable text features that, in turn, can help extract trading signals from extensive collections of texts. They speed up document review, enable the clustering of similar documents, and produce annotations useful for predictive modeling.
Applications include identifying critical themes in company disclosures, earnings call transcripts or contracts, and annotation based on sentiment analysis or using returns of related assets. 

<p align="center">
<img src="https://i.imgur.com/VVSnTCa.png" width="60%">
</p>

More specifically, it covers:
- How topic modeling has evolved, what it achieves, and why it matters
- Reducing the dimensionality of the DTM using latent semantic indexing
- Extracting topics with probabilistic latent semantic analysis (pLSA)
- How latent Dirichlet allocation (LDA) improves pLSA to become the most popular topic model
- Visualizing and evaluating topic modeling results -
- Running LDA using scikit-learn and gensim
- How to apply topic modeling to collections of earnings calls and financial news articles

### 16 Word embeddings for Earnings Calls and SEC Filings

This [chapter](16_word_embeddings) uses neural networks to learn a vector representation of individual semantic units like a word or a paragraph. These vectors are dense with a few hundred real-valued entries, compared to the higher-dimensional sparse vectors of the bag-of-words model. As a result, these vectors embed or locate each semantic unit in a continuous vector space.

Embeddings result from training a model to relate tokens to their context with the benefit that similar usage implies a similar vector. As a result, they encode semantic aspects like relationships among words through their relative location. They are powerful features that we will use with deep learning models in the following chapters.

<p align="center">
<img src="https://i.imgur.com/v8w9XLL.png" width="80%">
</p>

 More specifically, in this chapter, we will cover:
- What word embeddings are and how they capture semantic information
- How to obtain and use pre-trained word vectors
- Which network architectures are most effective at training word2vec models
- How to train a word2vec model using TensorFlow and gensim
- Visualizing and evaluating the quality of word vectors
- How to train a word2vec model on SEC filings to predict stock price moves
- How doc2vec extends word2vec and helps with sentiment analysis
- Why the transformer’s attention mechanism had such an impact on NLP
- How to fine-tune pre-trained BERT models on financial data

## Part 4: Deep & Reinforcement Learning

Part four explains and demonstrates how to leverage deep learning for algorithmic trading. 
The powerful capabilities of deep learning algorithms to identify patterns in unstructured data make it particularly suitable for alternative data like images and text. 

The sample applications show, for exapmle, how to combine text and price data to predict earnings surprises from SEC filings, generate synthetic time series to expand the amount of training data, and train a trading agent using deep reinforcement learning.
Several of these applications replicate research recently published in top journals.

### 17 Deep Learning for Trading

This [chapter](17_deep_learning) presents feedforward neural networks (NN) and demonstrates how to efficiently train large models using backpropagation while managing the risks of overfitting. It also shows how to use TensorFlow 2.0 and PyTorch and how to optimize a NN architecture to generate trading signals.
In the following chapters, we will build on this foundation to apply various architectures to different investment applications with a focus on alternative data. These include recurrent NN tailored to sequential data like time series or natural language and convolutional NN, particularly well suited to image data. We will also cover deep unsupervised learning, such as how to create synthetic data using Generative Adversarial Networks (GAN). Moreover, we will discuss reinforcement learning to train agents that interactively learn from their environment.

<p align="center">
<img src="https://i.imgur.com/5cet0Fi.png" width="70%">
</p>

In particular, this chapter will cover
- How DL solves AI challenges in complex domains
- Key innovations that have propelled DL to its current popularity
- How feedforward networks learn representations from data
- Designing and training deep neural networks (NNs) in Python
- Implementing deep NNs using Keras, TensorFlow, and PyTorch
- Building and tuning a deep NN to predict asset returns
- Designing and backtesting a trading strategy based on deep NN signals

### 18 CNN for Financial Time Series and Satellite Images

CNN architectures continue to evolve. This chapter describes building blocks common to successful applications, demonstrates how transfer learning can speed up learning, and how to use CNNs for object detection.
CNNs can generate trading signals from images or time-series data. Satellite data can anticipate commodity trends via aerial images of agricultural areas, mines, or transport networks. Camera footage can help predict consumer activity; we show how to build a CNN that classifies economic activity in satellite images.
CNNs can also deliver high-quality time-series classification results by exploiting their structural similarity with images, and we design a strategy based on time-series data formatted like images. 

<p align="center">
<img src="https://i.imgur.com/PlLQV0M.png" width="60%">
</p>

More specifically, this [chapter](18_convolutional_neural_nets) covers:

- How CNNs employ several building blocks to efficiently model grid-like data
- Training, tuning and regularizing CNNs for images and time series data using TensorFlow
- Using transfer learning to streamline CNNs, even with fewer data
- Designing a trading strategy using return predictions by a CNN trained on time-series data formatted like images
- How to classify economic activity based on satellite images

### 19 RNN for Multivariate Time Series and Sentiment Analysis

Recurrent neural networks (RNNs) compute each output as a function of the previous output and new data, effectively creating a model with memory that shares parameters across a deeper computational graph. Prominent architectures include Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) that address the challenges of learning long-range dependencies.
RNNs are designed to map one or more input sequences to one or more output sequences and are particularly well suited to natural language. They can also be applied to univariate and multivariate time series to predict market or fundamental data. This chapter covers how RNN can model alternative text data using the word embeddings that we covered in Chapter 16 to classify the sentiment expressed in documents.

<p align="center">
<img src="https://i.imgur.com/E9fOApg.png" width="60%">
</p>

More specifically, this chapter addresses:
- How recurrent connections allow RNNs to memorize patterns and model a hidden state
- Unrolling and analyzing the computational graph of RNNs
- How gated units learn to regulate RNN memory from data to enable long-range dependencies
- Designing and training RNNs for univariate and multivariate time series in Python
- How to learn word embeddings or use pretrained word vectors for sentiment analysis with RNNs
- Building a bidirectional RNN to predict stock returns using custom word embeddings

### 20 Autoencoders for Conditional Risk Factors and Asset Pricing

This [chapter](20_autoencoders_for_conditional_risk_factors) shows how to leverage unsupervised deep learning for trading. We also discuss autoencoders, namely, a neural network trained to reproduce the input while learning a new representation encoded by the parameters of a hidden layer. Autoencoders have long been used for nonlinear dimensionality reduction, leveraging the NN architectures we covered in the last three chapters.
We replicate a recent AQR paper that shows how autoencoders can underpin a trading strategy. We will use a deep neural network that relies on an autoencoder to extract risk factors and predict equity returns, conditioned on a range of equity attributes.

<p align="center">
<img src="https://i.imgur.com/aCmE0UD.png" width="60%">
</p>

More specifically, in this chapter you will learn about:
- Which types of autoencoders are of practical use and how they work
- Building and training autoencoders using Python
- Using autoencoders to extract data-driven risk factors that take into account asset characteristics to predict returns

### 21 Generative Adversarial Nets for Synthetic Time Series Data

This chapter introduces generative adversarial networks (GAN). GANs train a generator and a discriminator network in a competitive setting so that the generator learns to produce samples that the discriminator cannot distinguish from a given class of training data. The goal is to yield a generative model capable of producing synthetic samples representative of this class.
While most popular with image data, GANs have also been used to generate synthetic time-series data in the medical domain. Subsequent experiments with financial data explored whether GANs can produce alternative price trajectories useful for ML training or strategy backtests. We replicate the 2019 NeurIPS Time-Series GAN paper to illustrate the approach and demonstrate the results.

<p align="center">
<img src="https://i.imgur.com/W1Rp89K.png" width="60%">
</p>

More specifically, in this chapter you will learn about:
- How GANs work, why they are useful, and how they could be applied to trading
- Designing and training GANs using TensorFlow 2
- Generating synthetic financial data to expand the inputs available for training ML models and backtesting

### 22 Deep Reinforcement Learning: Building a Trading Agent

Reinforcement Learning (RL) models goal-directed learning by an agent that interacts with a stochastic environment. RL optimizes the agent's decisions concerning a long-term objective by learning the value of states and actions from a reward signal. The ultimate goal is to derive a policy that encodes behavioral rules and maps states to actions.
This [chapter](22_deep_reinforcement_learning) shows how to formulate and solve an RL problem. It covers model-based and model-free methods, introduces the OpenAI Gym environment, and combines deep learning with RL to train an agent that navigates a complex environment. Finally, we'll show you how to adapt RL to algorithmic trading by modeling an agent that interacts with the financial market while trying to optimize an objective function.

<p align="center">
<img src="https://i.imgur.com/lg0ofbZ.png" width="60%">
</p>

More specifically,this chapter will cover:

- Define a Markov decision problem (MDP)
- Use value and policy iteration to solve an MDP
- Apply Q-learning in an environment with discrete states and actions
- Build and train a deep Q-learning agent in a continuous environment
- Use the OpenAI Gym to design a custom market environment and train an RL agent to trade stocks

### 23 Conclusions and Next Steps

In this concluding chapter, we will briefly summarize the essential tools, applications, and lessons learned throughout the book to avoid losing sight of the big picture after so much detail.
We will then identify areas that we did not cover but would be worth focusing on as you expand on the many machine learning techniques we introduced and become productive in their daily use.

In sum, in this chapter, we will
- Review key takeaways and lessons learned
- Point out the next steps to build on the techniques in this book
- Suggest ways to incorporate ML into your investment process

### 24 Appendix - Alpha Factor Library

Throughout this book, we emphasized how the smart design of features, including appropriate preprocessing and denoising, typically leads to an effective strategy. This appendix synthesizes some of the lessons learned on feature engineering and provides additional information on this vital topic.

To this end, we focus on the broad range of indicators implemented by TA-Lib (see [Chapter 4](04_alpha_factor_research)) and WorldQuant's [101 Formulaic Alphas](https://arxiv.org/pdf/1601.00991.pdf) paper (Kakushadze 2016), which presents real-life quantitative trading factors used in production with an average holding period of 0.6-6.4 days.

This chapter covers: 
- How to compute several dozen technical indicators using TA-Lib and NumPy/pandas,
- Creating the formulaic alphas describe in the above paper, and
- Evaluating the predictive quality of the results using various metrics from rank correlation and mutual information to feature importance, SHAP values and Alphalens.

- 交易机器学习 -第二版
本书旨在展示机器学习如何以实用而全面的方式为算法交易策略增加价值。它涵盖了从线性回归到深度强化学习的广泛机器学习技术，并演示了如何构建、回测和评估由模型预测驱动的交易策略。

它分为四部分，共23 章和一个附录，共800 多页：

数据源、金融特征工程和投资组合管理的重要方面，
基于监督和无监督机器学习算法的多空策略的设计和评估，
如何从SEC 文件、财报电话会议记录或财经新闻等金融文本数据中提取可交易信号，
将 CNN 和 RNN 等深度学习模型与市场和替代数据结合使用，如何通过生成对抗网络生成合成数据，以及使用深度强化学习训练交易代理


该存储库包含150 多个笔记本，它们将书中讨论的概念、算法和用例付诸实践。他们提供了大量的例子表明：

如何使用市场、基本面和替代文本和图像数据并从中提取信号，
如何训练和调整模型来预测不同资产类别和投资期限的回报，包括如何复制最近发表的研究，以及
如何设计、回测和评估交易策略。
我们强烈建议您在阅读本书时查看笔记本；它们通常处于已执行状态，并且通常包含由于空间限制而未包含的附加信息。

除了本存储库中的信息之外，本书的网站还包含章节摘要和其他信息。

加入 ML4T 社区！
为了方便读者就本书的内容和代码示例以及自己的策略和行业发展的制定和实施提出问题，我们正在托管一个在线平台。

请加入我们的社区，与有兴趣利用机器学习进行交易策略的其他交易者联系，分享您的经验，并互相学习！

第二版有什么新内容？
首先，本书演示了如何使用各种监督、无监督和强化学习算法从各种数据源中提取信号并为不同资产类别设计交易策略。它还提供相关的数学和统计知识，以方便算法的调整或结果的解释。此外，它还涵盖了金融背景，可帮助您处理市场和基本数据、提取信息特征以及管理交易策略的绩效。

从实践的角度来看，第二版旨在为您提供概念理解和工具，以开发您自己的基于机器学习的交易策略。为此，它将机器学习视为流程中的关键元素，而不是独立的练习，引入了从数据源、特征工程、模型优化到策略设计和回测的交易工作流程的端到端机器学习。

更具体地说，ML4T 工作流程首先为明确定义的投资领域产生想法，收集相关数据并提取信息特征。它还涉及设计、调整和评估适合预测任务的机器学习模型。最后，它需要开发交易策略来作用于模型的预测信号，并使用回测引擎模拟和评估其在历史数据上的表现。一旦您决定在真实市场中执行算法策略，您会发现自己反复迭代此工作流程以融入新信息和不断变化的环境。



第二版对 ML4t 工作流程的强调转化为关于策略回测的新章节、描述 100 多个不同 alpha 因子的新附录以及许多新的实际应用。为了清晰和可读性，我们还重写了大部分现有内容。

交易应用程序现在使用除每日美国股票价格之外更广泛的数据源，包括国际股票和 ETF。它还演示了如何将机器学习用于具有分钟频率股票数据的日内策略。此外，它还扩展了替代数据源的覆盖范围，包括用于情绪分析和回报预测的 SEC 文件，以及用于对土地用途进行分类的卫星图像。

第二版的另一个创新是复制了最近在顶级期刊上发表的几个交易应用程序：

第 18 章演示了如何将卷积神经网络应用于转换为图像格式的时间序列，以基于Sezer 和 Ozbahoglu（2018）进行回报预测。
第 20 章展示了如何使用基于 Shihao Gu、Bryan T. Kelly 和 Da Cheng Xiu (2019) 的自动编码器资产定价模型的自动编码器提取以股票特征为条件的风险因素，以进行资产定价，以及
第 21 章展示了如何使用基于时间序列生成对抗网络的生成对抗网络创建合成训练数据，作者是 Jinsung Yoon、Daniel Jarrett 和 Mihaela van der Schaar（2019 年）。
所有应用程序现在都使用最新的可用（在撰写本文时）软件版本，例如 pandas 1.0 和 TensorFlow 2.2。还有一个定制版本的 Zipline，可以在设计交易策略时轻松包含机器学习模型预测。

安装、数据源和错误报告
这些代码示例依赖于数据科学和金融领域的各种 Python 库。

没有必要尝试一次安装所有库，因为这会增加遇到版本冲突的可能性。相反，我们建议您在阅读过程中安装特定章节所需的库。

2022 年 3 月更新：zipline-reloaded、pyfolio-reloaded、alphalens-reloaded和empyrical-reloaded现已在conda-forge频道上提供。该频道ml4t仅包含过时的版本，很快就会被删除。

2021年4月更新：随着Zipline的更新，不再需要使用Docker。安装说明现在引用了特定于操作系统的环境文件，这些文件应该可以简化笔记本电脑的运行。

2021 年 2 月更新：代码示例版本 2.0 将 Docker 映像提供的 conda 环境更新为 Python 3.8、Pandas 1.2 和 TensorFlow 1.2 等；Zipline 回测环境现在使用 Python 3.6。

安装目录包含有关设置和使用 Docker 映像来运行笔记本的详细说明。它还包含用于设置各种conda环境的配置文件，如果您愿意，可以直接在您的计算机上安装笔记本中使用的软件包（并且，根据您的系统，准备好加倍努力）。
要下载和预处理本书中使用的许多数据源，请参阅自述文件中的说明以及数据目录中的各种笔记本。
如果您在安装环境、下载数据或运行代码时遇到任何困难，请在存储库中提出GitHub 问题（此处）。此处描述了如何处理 GitHub 问题。

更新：您可以在此处下载本书中使用的algoseek数据。请参阅第 2 章中的预处理说明和第 12 章中的梯度增强模型的日内示例。

更新：图形目录包含书中使用的图表的彩色版本。

大纲和章节摘要
本书分为四个部分，分别解决在获取和使用市场、基本数据和替代数据源时出现的不同挑战，为交易环境中的各种预测任务开发机器学习解决方案，以及设计和评估依赖于生成的预测信号的交易策略。机器学习模型。

每章的目录都包含一个自述文件，其中包含有关内容、代码示例和其他资源的附加信息。

第 1 部分：从数据到战略制定

01 交易机器学习：从想法到执行
02 市场与基本数据：来源和技术
03 金融替代数据：类别和用例
04 金融特征工程：如何研究Alpha因子
05 投资组合优化与绩效评估
第 2 部分：交易机器学习：基础知识

06 机器学习过程
07 线性模型：从风险因素到回报预测
08 ML4T 工作流程：从模型到策略回测
09 波动率预测和统计套利的时间序列模型
10 贝叶斯机器学习：动态夏普比率和配对交易
11 随机森林：日本股票的多空策略
12 提升您的交易策略
13 数据驱动的风险因素和无监督学习的资产配置
第 3 部分：交易自然语言处理

14 交易文本数据：情绪分析
15 主题建模：总结财经新闻
16 财报电话会议和 SEC 文件的词嵌入
第 4 部分：深度学习和强化学习

17 深度学习交易
18 CNN 金融时间序列和卫星图像
19 用于多元时间序列和情感分析的 RNN
20 个用于条件风险因素和资产定价的自动编码器
21 用于合成时间序列数据的生成对抗网络
22 深度强化学习：构建交易代理
23 结论和后续步骤
24 附录 - Alpha 因子库
第 1 部分：从数据到战略制定
第一部分提供了一个用于开发由机器学习 (ML) 驱动的交易策略的框架。它重点关注为本书中讨论的机器学习算法和策略提供支持的数据，概述了如何设计和评估适合机器学习模型的功能，以及如何在执行交易策略时管理和衡量投资组合的表现。

01 交易机器学习：从想法到执行
本章探讨了导致机器学习成为投资行业竞争优势来源的行业趋势。我们还将研究机器学习在投资流程中的应用，以实现算法交易策略。

更具体地说，它涵盖以下主题：

机器学习在投资行业崛起背后的主要趋势
利用机器学习的交易策略的设计和执行
机器学习在交易中的流行用例
02 市场与基本数据：来源和技术
本章展示如何使用市场和基本数据，并描述它们所反映的环境的关键方面。例如，熟悉各种订单类型和交易基础设施不仅对于数据的解释很重要，而且对于正确设计回测模拟也很重要。我们还说明了如何使用 Python 访问和操作交易和财务报表数据。

实际示例演示了如何使用来自 NASDAQ 逐笔报价数据和 Algoseek 分钟柱数据的交易数据，以及一组丰富的属性来捕获供需动态，我们稍后将使用这些属性来创建基于 ML 的日内策略。我们还介绍了各种数据提供商 API 以及如何从 SEC 获取财务报表信息。

订单簿

本章特别涵盖：
市场数据如何反映交易环境的结构
以分钟频率处理日内交易和报价数据
使用 NASDAQ ITCH 从逐笔报价数据重建限价订单簿
使用各种类型的柱形图汇总刻度数据
使用可扩展业务报告语言 (XBRL) 编码的电子申报
解析并结合市场和基本数据以创建市盈率系列
如何使用Python访问各种市场和基本数据源
03 金融替代数据：类别和用例
本章概述了替代数据的类别和用例，描述了评估爆炸式增长的数据源和提供商的标准，并总结了当前的市场格局。

它还演示了如何通过抓取网站来创建替代数据集，例如收集财报电话会议记录以与本书第三部分中的自然语言处理（NLP）和情绪分析算法一起使用。

更具体地说，本章涵盖：

另类数据革命期间出现了哪些新信号源
个人、企业和传感器如何生成多样化的替代数据
另类数据的重要类别和提供者
评估如何将不断增长的替代数据用于交易
在 Python 中使用替代数据，例如通过抓取互联网
04 金融特征工程：如何研究Alpha因子
如果您已经熟悉 ML，您就会知道特征工程是成功预测的关键因素。它在交易领域至少同样重要，学术界和行业研究人员几十年来一直在研究驱动资产市场和价格的因素，以及哪些特征有助于解释或预测价格变动。



本章概述了这项研究的主要要点，作为您自己探索阿尔法因子的起点。它还提供了计算和测试 alpha 因子的基本工具，重点介绍了 NumPy、pandas 和 TA-Lib 库如何促进数据操作，并介绍了小波和卡尔曼滤波器等流行的平滑技术，有助于减少数据中的噪声。读完后，您将了解：

存在哪些类别的因素、它们为何起作用以及如何衡量它们，
使用 NumPy、pandas 和 TA-Lib 创建 alpha 因子，
如何使用小波和卡尔曼滤波器对数据进行去噪，
使用 Zipline 测试单个和多个 alpha 因子，
如何使用Alphalens评估预测性能。
05 投资组合优化与绩效评估
Alpha 因子产生信号，将算法策略转化为交易，进而产生多头和空头头寸。由此产生的投资组合的回报和风险决定了该策略是否满足投资目标。



有多种方法可以优化投资组合。其中包括应用机器学习（ML）来学习资产之间的层次关系，并在设计投资组合的风险状况时将它们视为补充或替代品。本章内容包括：

如何衡量投资组合的风险和回报
使用均值方差优化和替代方案管理投资组合权重
使用机器学习在投资组合环境中优化资产配置
使用 Zipline 模拟交易并根据 alpha 因子创建投资组合
如何使用pyfolio评估投资组合绩效
第 2 部分：交易机器学习：基础知识
第二部分涵盖了基本的监督和无监督学习算法，并说明了它们在交易策略中的应用。它还介绍了 Quantopian 平台，允许您利用和结合本书中开发的数据和机器学习技术来实施在实时市场中执行交易的算法策略。

06 机器学习过程
本章是第 2 部分的开始，第 2部分说明了如何使用一系列监督和无监督的 ML 模型进行交易。在使用各种 Python 库演示相关应用程序之前，我们将解释每个模型的假设和用例。

许多这些模型及其应用程序有几个共同点。本章涵盖了这些常见方面，以便我们可以在后续章节中重点关注特定于模型的用法。它通过概述如何将 ML 模型的预测性能制定、训练、调整和评估作为系统工作流程奠定了基础。内容包括：



监督和无监督数据学习如何运作
训练和评估回归和分类任务的监督学习模型
偏差-方差权衡如何影响预测性能
如何诊断和解决由于过度拟合而导致的预测错误
使用交叉验证来优化超参数，重点关注时间序列数据
为什么在样本外测试时需要额外关注财务数据
07 线性模型：从风险因素到回报预测
线性模型是回归和分类环境中推理和预测的标准工具。许多广泛使用的资产定价模型都依赖于线性回归。Ridge 和 Lasso 回归等正则化模型通常可以通过限制过度拟合的风险来产生更好的预测。典型的回归应用程序识别驱动资产回报的风险因素，以管理风险或预测回报。另一方面，分类问题包括定向价格预测。



第 07 章涵盖以下主题：

线性回归如何工作以及它做出哪些假设
训练和诊断线性回归模型
使用线性回归预测股票收益
使用正则化来提高预测性能
逻辑回归的工作原理
将回归问题转化为分类问题
08 ML4T 工作流程：从模型到策略回测
本章从端到端的角度介绍了设计、模拟和评估由机器学习算法驱动的交易策略。我们将详细演示如何使用 Python 库backtrader和Zipline在历史市场环境中回溯测试 ML 驱动的策略。ML4T 工作流程的最终目标是从历史数据中收集证据，帮助决定是否在实时市场中部署候选策略并将财务资源置于风险之中。对策略的真实模拟需要忠实地反映证券市场的运作方式以及交易的执行方式。此外，需要注意几个方法方面的问题，以避免有偏见的结果和错误的发现，从而导致错误的投资决策。



更具体地说，学完本章后，您将能够：

规划并实施端到端策略回测
了解并避免实施回测时的关键陷阱
讨论矢量化回测引擎与事件驱动回测引擎的优缺点
识别并评估事件驱动回测器的关键组件
使用数据源以分钟和每日的频率设计和执行 ML4T 工作流程，并单独训练 ML 模型或作为回测的一部分
使用 Zipline 和 backtrader 设计和评估您自己的策略
09 波动率预测和统计套利的时间序列模型
本章重点介绍从时间序列历史中提取信号以预测同一时间序列的未来值的模型。由于交易固有的时间维度，时间序列模型被广泛使用。它提供了诊断时间序列特征（例如平稳性）和提取捕获潜在有用模式的特征的工具。它还引入了单变量和多元时间序列模型来预测宏观数据和波动模式。最后，它解释了协整如何识别时间序列中的共同趋势，并展示了如何基于这一关键概念开发配对交易策略。



特别是，它涵盖：

如何使用时间序列分析来准备建模过程并为其提供信息
估计和诊断单变量自回归和移动平均模型
构建自回归条件异方差 (ARCH) 模型来预测波动性
如何构建多元向量自回归模型
使用协整开发配对交易策略
10 贝叶斯机器学习：动态夏普比率和配对交易
贝叶斯统计使我们能够量化未来事件的不确定性，并在新信息到来时以有原则的方式完善估计。这种动态方法很好地适应了金融市场不断变化的性质。机器学习的贝叶斯方法能够对统计指标、参数估计和预测的不确定性提供新的见解。应用范围从更精细的风险管理到包含市场环境变化的预测模型的动态更新。



更具体地说，本章涵盖：

贝叶斯统计如何应用于机器学习
使用 PyMC3 进行概率编程
使用 PyMC3 定义和训练机器学习模型
如何运行最先进的采样方法来进行近似推理
贝叶斯机器学习应用程序，用于计算动态夏普比率、动态货币对交易对冲比率以及估计随机波动率
11 随机森林：日本股票的多空策略
本章将决策树和随机森林应用于交易。决策树从编码非线性输入输出关系的数据中学习规则。我们展示了如何训练决策树来预测回归和分类问题，可视化和解释模型学到的规则，以及调整模型的超参数以优化偏差-方差权衡并防止过度拟合。

本章的第二部分介绍了集成模型，该模型以随机方式组合多个决策树，以产生误差较低的单个预测。最后得出了基于随机森林模型生成的交易信号的日本股票多空策略。



简而言之，本章涵盖：

使用决策树进行回归和分类
从决策树中获取见解并可视化从数据中学到的规则
了解为什么集成模型往往会提供出色的结果
使用引导聚合来解决决策树的过度拟合挑战
训练、调整和解释随机森林
采用随机森林来设计和评估有利可图的交易策略
12 提升您的交易策略
梯度增强是一种替代的基于树的集成算法，通常比随机森林产生更好的结果。关键的区别在于，Boosting 根据模型产生的累积误差修改用于训练每棵树的数据。虽然随机森林使用数据的随机子集独立地训练许多树，但提升会按顺序进行并重新加权数据。本章展示了最先进的库如何实现令人印象深刻的性能，并将提升应用于每日和高频数据来回测日内交易策略。



更具体地说，我们将涵盖以下主题：

boosting 与 bagging 有何不同，梯度 boosting 是如何从自适应 boosting 演变而来的，
使用 scikit-learn 设计和调整自适应和梯度增强模型，
使用最先进的实现 XGBoost、LightGBM 和 CatBoost 在大型数据集上构建、优化和评估梯度增强模型，
使用SHA值解释梯度增强模型并从中获得见解，以及
使用高频数据的提升来设计日内策略。
13 数据驱动的风险因素和无监督学习的资产配置
降维和聚类是无监督学习的主要任务：

降维将现有特征转换为新的、更小的集合，同时最大限度地减少信息损失。存在多种算法，这些算法的不同之处在于它们如何衡量信息丢失，是否应用线性或非线性变换，或者它们对新特征集施加的约束。
聚类算法识别相似的观察或特征并将其分组，而不是识别新特征。算法的不同之处在于如何定义观察的相似性以及对结果组的假设。


更具体地说，本章涵盖：

主成分分析和独立成分分析（PCA 和 ICA）如何执行线性降维
使用 PCA 从资产回报中识别数据驱动的风险因素和特征投资组合
使用流形学习有效地可视化非线性、高维数据
使用T-SNE和UMAP探索高维图像数据
k 均值、分层和基于密度的聚类算法如何工作
使用凝聚聚类构建具有分层风险平价的稳健投资组合
第 3 部分：交易自然语言处理
文本数据内容丰富，但格式非结构化，因此需要更多预处理，以便机器学习算法能够提取潜在信号。关键的挑战包括将文本转换为可供算法使用的数字格式，同时表达内容的语义或含义。

接下来的三章介绍了几种捕捉人类容易理解的语言细微差别的技术，以便机器学习算法也可以解释它们。

14 交易文本数据：情绪分析
文本数据内容非常丰富，但高度非结构化，因此需要更多的预处理才能使机器学习算法能够提取相关信息。一个关键的挑战包括将文本转换为数字格式而不失去其含义。本章介绍如何通过创建文档术语矩阵将文档表示为标记计数向量，该矩阵又充当文本分类和情感分析的输入。它还介绍了朴素贝叶斯算法，并将其性能与线性和基于树的模型进行了比较。

特别是，本章涵盖：

NLP 的基本工作流程是什么样的
如何使用 spaCy 和 TextBlob 构建多语言特征提取管道
执行 NLP 任务，例如词性标记或命名实体识别
使用文档术语矩阵将标记转换为数字
使用朴素贝叶斯模型对新闻进行分类
如何使用不同的机器学习算法执行情感分析
15 主题建模：总结财经新闻
本章使用无监督学习来建模潜在主题并从文档中提取隐藏主题。这些主题可以生成对大量财务报告的详细见解。主题模型可以自动创建复杂的、可解释的文本特征，从而有助于从大量文本集合中提取交易信号。它们加快文档审查速度，实现相似文档的聚类，并生成对预测建模有用的注释。应用包括识别公司披露、财报电话会议记录或合同中的关键主题，以及基于情绪分析或使用相关资产回报的注释。



更具体地说，它涵盖：

主题建模如何演变、它取得了什么成果以及它为何重要
使用潜在语义索引降低 DTM 的维度
使用概率潜在语义分析 (pLSA) 提取主题
潜在狄利克雷分配（LDA）如何改进 pLSA 成为最流行的主题模型
可视化和评估主题建模结果 -
使用 scikit-learn 和 gensim 运行 LDA
如何将主题建模应用于财报电话会议和财经新闻文章的集合
16 财报电话会议和 SEC 文件的词嵌入
本章使用神经网络来学习单个语义单元（例如单词或段落）的向量表示。与词袋模型的高维稀疏向量相比，这些向量密集，包含数百个实值条目。结果，这些向量将每个语义单元嵌入或定位在连续向量空间中。

嵌入是通过训练模型将标记与其上下文相关联而产生的，其优点是相似的用法意味着相似的向量。因此，它们通过单词之间的相对位置来编码语义方面，例如单词之间的关系。它们是我们将在接下来的章节中与深度学习模型一起使用的强大功能。



更具体地说，在本章中，我们将介绍：

什么是词嵌入以及它们如何捕获语义信息
如何获取并使用预训练的词向量
哪些网络架构在训练 word2vec 模型方面最有效
如何使用 TensorFlow 和 gensim 训练 word2vec 模型
可视化和评估词向量的质量
如何根据 SEC 文件训练 word2vec 模型来预测股价走势
doc2vec 如何扩展 word2vec 并帮助进行情感分析
为什么 Transformer 的注意力机制对 NLP 产生如此大的影响
如何根据金融数据微调预训练的 BERT 模型
第 4 部分：深度学习和强化学习
第四部分解释并演示了如何利用深度学习进行算法交易。深度学习算法识别非结构化数据模式的强大功能使其特别适合图像和文本等替代数据。

例如，示例应用程序展示了如何结合文本和价格数据来预测 SEC 文件中的收益意外，生成合成时间序列以扩大训练数据量，以及使用深度强化学习来训练交易代理。其中一些应用复制了最近在顶级期刊上发表的研究。

17 深度学习交易
本章_介绍前馈神经网络 (NN)，并演示如何使用反向传播有效地训练大型模型，同时管理过度拟合的风险。它还展示了如何使用 TensorFlow 2.0 和 PyTorch 以及如何优化神经网络架构来生成交易信号。在接下来的章节中，我们将在此基础上将各种架构应用于不同的投资应用程序，重点关注另类数据。其中包括针对时间序列或自然语言等序列数据定制的循环神经网络，以及特别适合图像数据的卷积神经网络。我们还将介绍深度无监督学习，例如如何使用生成对抗网络（GAN）创建合成数据。此外，我们将讨论强化学习来训练从环境中交互式学习的代理。



特别是，本章将涵盖

深度学习如何解决复杂领域的人工智能挑战
推动深度学习达到目前流行程度的关键创新
前馈网络如何从数据中学习表示
使用 Python 设计和训练深度神经网络 (NN)
使用 Keras、TensorFlow 和 PyTorch 实现深度神经网络
构建和调整深度神经网络来预测资产回报
基于深度神经网络信号设计和回测交易策略
18 CNN 金融时间序列和卫星图像
CNN 架构不断发展。本章介绍了成功应用程序中常见的构建块，演示了迁移学习如何加速学习，以及如何使用 CNN 进行对象检测。CNN 可以从图像或时间序列数据生成交易信号。卫星数据可以通过农业区、矿山或运输网络的航空图像来预测商品趋势。摄像机镜头可以帮助预测消费者活动；我们展示了如何构建一个 CNN，对卫星图像中的经济活动进行分类。CNN 还可以利用其与图像的结构相似性来提供高质量的时间序列分类结果，我们设计了一种基于图像格式的时间序列数据的策略。



更具体地说，本章涵盖：

CNN 如何使用多个构建块对网格状数据进行高效建模
使用 TensorFlow 针对图像和时间序列数据训练、调整和正则化 CNN
使用迁移学习来简化 CNN，即使数据较少
使用基于图像格式的时间序列数据训练的 CNN 进行回报预测来设计交易策略
如何根据卫星图像对经济活动进行分类
19 用于多元时间序列和情感分析的 RNN
循环神经网络 (RNN) 将每个输出计算为先前输出和新数据的函数，从而有效地创建一个具有内存的模型，该模型在更深的计算图中共享参数。著名的架构包括长短期记忆 (LSTM) 和门控循环单元 (GRU)，它们解决了学习长期依赖关系的挑战。RNN 旨在将一个或多个输入序列映射到一个或多个输出序列，特别适合自然语言。它们还可以应用于单变量和多元时间序列来预测市场或基本数据。本章介绍 RNN 如何使用第 16 章中介绍的词嵌入对替代文本数据进行建模，以对文档中表达的情感进行分类。



更具体地说，本章讨论：

循环连接如何让 RNN 记住模式并对隐藏状态建模
展开并分析 RNN 的计算图
门控单元如何学习根据数据调节 RNN 内存以实现远程依赖
使用 Python 设计和训练单变量和多变量时间序列的 RNN
如何学习词嵌入或使用预训练的词向量通过 RNN 进行情感分析
使用自定义词嵌入构建双向 RNN 来预测股票收益
20 个用于条件风险因素和资产定价的自动编码器
本章展示如何利用无监督深度学习进行交易。我们还讨论了自动编码器，即一种经过训练可以再现输入的神经网络，同时学习由隐藏层参数编码的新表示。自动编码器长期以来一直用于非线性降维，利用我们在过去三章中介绍的神经网络架构。我们复制了最近的 AQR 论文，该论文展示了自动编码器如何支持交易策略。我们将使用依赖于自动编码器的深度神经网络来提取风险因素并预测股票回报，以一系列股票属性为条件。



更具体地说，在本章中您将了解：

哪些类型的自动编码器具有实际用途以及它们如何工作
使用 Python 构建和训练自动编码器
使用自动编码器提取数据驱动的风险因素，考虑资产特征来预测回报
21 用于合成时间序列数据的生成对抗网络
本章介绍生成对抗网络（GAN）。GAN 在竞争环境中训练生成器和鉴别器网络，以便生成器学习生成鉴别器无法与给定类别的训练数据区分开来的样本。目标是产生一个能够生成代表此类的合成样本的生成模型。虽然 GAN 在图像数据中最受欢迎，但它也被用来在医学领域生成合成时间序列数据。随后的金融数据实验探讨了 GAN 是否可以产生可用于 ML 训练或策略回测的替代价格轨迹。我们复制 2019 年 NeurIPS 时间序列 GAN 论文来说明该方法并展示结果。



更具体地说，在本章中您将了解：

GAN 的工作原理、它们为何有用以及它们如何应用于交易
使用 TensorFlow 2 设计和训练 GAN
生成综合财务数据以扩展可用于训练 ML 模型和回溯测试的输入
22 深度强化学习：构建交易代理
强化学习 (RL) 通过与随机环境交互的代理来模拟目标导向的学习。强化学习通过从奖励信号中学习状态和动作的价值来优化代理关于长期目标的决策。最终目标是导出一种对行为规则进行编码并将状态映射到行动的策略。本章展示如何制定和解决强化学习问题。它涵盖了基于模型和无模型的方法，介绍了 OpenAI Gym 环境，并将深度学习与 RL 相结合来训练在复杂环境中导航的代理。最后，我们将向您展示如何通过对与金融市场交互的代理进行建模，同时尝试优化目标函数，使强化学习适应算法交易。



更具体地说，本章将涵盖：

定义马尔可夫决策问题 (MDP)
使用价值和策略迭代来解决 MDP
在具有离散状态和动作的环境中应用 Q-learning
在连续环境中构建和训练深度 Q 学习代理
使用 OpenAI Gym 设计自定义市场环境并训练 RL 代理来交易股票
23 结论和后续步骤
在最后一章中，我们将简要总结整本书中的基本工具、应用程序和经验教训，以避免在这么多细节之后忽视全局。然后，当您扩展我们介绍的许多机器学习技术并在日常使用中变得富有成效时，我们将确定我们没有涵盖但值得关注的领域。

总而言之，在本章中，我们将

回顾要点和经验教训
指出基于本书中的技术的后续步骤
提出将机器学习纳入投资流程的方法建议
24 附录 - Alpha 因子库
在本书中，我们强调了特征的智能设计（包括适当的预处理和去噪）通常如何产生有效的策略。本附录综合了特征工程方面的一些经验教训，并提供了有关这一重要主题的更多信息。

为此，我们重点关注 TA-Lib（参见第 4 章）和 WorldQuant 的101 Formulaic Alphas论文（Kakushadze 2016）实施的广泛指标，该论文提出了生产中使用的真实量化交易因子，平均持有期为0.6-6.4天。

本章内容包括：

如何使用 TA-Lib 和 NumPy/pandas 计算几十个技术指标，
创建上述论文中描述的公式化 alpha，以及
使用从排名相关性和互信息到特征重要性、SHAP 值和 Alphalens 的各种指标来评估结果的预测质量。
