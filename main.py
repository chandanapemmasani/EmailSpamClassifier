#!/usr/bin/env python
# coding: utf-8

# ## Implementation of Spam classifier using Multinomial Naivebayes

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import re

# In[2]:


spamDataset = pd.read_csv('spamText.csv', sep=',', header=None, names=['Label', 'SMS'])
hamCount, spamCount = spamDataset['Label'].value_counts(normalize=True)

data = [hamCount, spamCount]
msgs = ['Ham messages', 'Spam messages']

plt.title("Distribution of class labels in complete dataset")

plt.pie(data, labels=msgs)

plt.show()

print("size of dataset:", spamDataset.shape)

# ## Splitting dataset into train and test datasets in the ratio 3:1

# In[3]:


trainTestRatio = round(len(spamDataset) * 0.75)

# In[4]:


trainSet = spamDataset[:trainTestRatio].reset_index(drop=True)

hamCount, spamCount = trainSet['Label'].value_counts(normalize=True)

data = [hamCount, spamCount]
msgs = ['Ham messages', 'Spam messages']

plt.title("Distribution of class labels in train dataset")

plt.pie(data, labels=msgs)

plt.show()

print("size of train dataSet:", trainSet.shape)

# In[5]:


testSet = spamDataset[trainTestRatio:].reset_index(drop=True)

hamCount, spamCount = testSet['Label'].value_counts(normalize=True)

data = [hamCount, spamCount]
msgs = ['Ham messages', 'Spam messages']

plt.title("Distribution of class labels in test dataset")

plt.pie(data, labels=msgs)

plt.show()

print("size of test dataSet:", testSet.shape)

# ## Data cleaning

# In[6]:


trainSet['SMS'] = trainSet['SMS'].astype(str).str.replace(
    '\W', ' ')  # removes punctuations
trainSet['SMS'] = trainSet['SMS'].str.lower()  # converts into lowercase

trainSet['SMS'] = trainSet['SMS'].str.split()  # splits the sentences into words

# ## Calculating total number of unique words in the training dataset

# In[7]:


totalWords = []
for msg in trainSet['SMS']:
    for word in msg:
        totalWords.append(word)

totalWords = list(set(totalWords))  # removes duplicates

# ## Transforming the dataset into dictionary that maintains count of each unique word for every message in the dataset.

# In[8]:


wordCountPerMsg = {unique_word: [0] * len(trainSet['SMS']) for unique_word in totalWords}
for index, msg in enumerate(trainSet['SMS']):
    for word in msg:
        wordCountPerMsg[word][index] += 1

wordCounts = pd.DataFrame(wordCountPerMsg)
trainDatareqFormat = pd.concat([trainSet, wordCounts], axis=1)
display(trainDatareqFormat.head())

# ## separating spam and ham messages from the training dataset

# In[9]:


spamMsgs = trainDatareqFormat[trainDatareqFormat['Label'] == 'spam']

hamMsgs = trainDatareqFormat[trainDatareqFormat['Label'] == 'ham']

spamProbability = len(spamMsgs) / len(trainDatareqFormat)  # calculating the probability of message being spam

hamProbability = len(hamMsgs) / len(trainDatareqFormat)  # calculating the probability of message being ham

wordCountperSpamMsg = spamMsgs['SMS'].apply(len)
totalSpamwordsCount = wordCountperSpamMsg.sum()  # finds total number of words in all spam messages

wordCountperHamMsg = hamMsgs['SMS'].apply(len)
totalHamwordsCount = wordCountperHamMsg.sum()  # finds total number of words in all ham messages

# ### When the algorithm finds a new word which is not present in train dataset, it assigns zero probability for that word as the count for that word would be zero. But this results in getting overall probability of the message being spam or ham zero. To avoid this we start the count from 1 instead of zero.
# ## This technique is called Laplace smoothing

# In[10]:


laplaceFac = 1

# # Naive Bayes algorithm implementation

# ### Probability of a message being spam given words {X1, X2, X3....Xn} = P(S|X1, X2,....,Xn)
# ### From Bayes theorem, we know that
# ####         P(S|X1, X2,....,Xn) = P(X1, X2, X3...,Xn|S) * P(S)/P(X1, X2, X3...,Xn)
# ####                                      = P(X1, X2, X3...,Xn|S) * P(S)/P(X1, X2, X3...,Xn|S) * P(S) + P(X1, X2, X3...,Xn|H) * P(H)
#
# ### ignoring the denominator for the moment and by the definition of conditional probability,
# ####         P(X1, X2, X3...,Xn|S) * P(S) =  P(X1, X2, X3...,Xn, S)
# ### using chain rule to decompose,
# ####         P(X1, X2, X3...,Xn, S) = P(X1| X2, X3...,Xn, S) * P(X2| X3, X4...,Xn, S).......*P(Xn-1| Xn, S) * P(Xn|S) * P(S)
# ### Naivebayes assumes conditional independence between these words and thus by assuming the words in the email are conditionally independent
# ####        P(X1, X2, X3...,Xn, S) ~ P(X1|S) * P(X2|S)......P(Xn|S)*P(S)
# ####                                            = P(S) $\prod_{i=1}^{i=n}$ P(Xi|S)
# ### similarly for a ham message,
# ####        P(X1, X2, X3...,Xn, H) ~ P(X1|H) * P(X2|H)......P(Xn|H)*P(H)
# ####                                            = P(H) $\prod_{i=1}^{i=n}$ P(Xi|H)

# In[11]:


# initialising probability parameters
prob_spam = {unique_word: 0 for unique_word in totalWords}
prob_ham = {unique_word: 0 for unique_word in totalWords}

for word in totalWords:
    wordCountgivenSpam = spamMsgs[word].sum()
    wordProbgivenSpam = (wordCountgivenSpam + laplaceFac) / (totalSpamwordsCount + laplaceFac)
    prob_spam[word] = wordProbgivenSpam  # storing P(word|spam) for all the words in spam

    wordCountgivenHam = hamMsgs[word].sum()
    wordProbgivenHam = (wordCountgivenHam + laplaceFac) / (totalHamwordsCount + laplaceFac)
    prob_ham[word] = wordProbgivenHam  # storing P(word|ham) for all the words in ham


# In[12]:


def classifyTestData(message):
    message = re.sub('\W', ' ', message)
    message = message.lower().split()

    spamProbGivenText = spamProbability  # P(S)
    hamProbGivenText = hamProbability  # P(H)

    for word in message:
        if word in prob_spam:
            spamProbGivenText *= prob_spam[word]  # P(S) * (P(word1|spam) * P(word2|spam)...P(wordn|spam))

        if word in prob_ham:
            hamProbGivenText *= prob_ham[word]  # P(H) * (P(word1|ham) * P(word2|ham)...P(wordn|ham))

    if hamProbGivenText > spamProbGivenText:
        return 'ham'
    elif spamProbGivenText > hamProbGivenText:
        return 'spam'
    else:
        return 'equal probabilities'


# In[13]:


predicted = []

falseNegative = 0
falsePositive = 0

for msg in testSet['SMS']:
    res = classifyTestData(msg)
    predicted.append(res)

testSet['predicted'] = predicted

correct = 0
total = testSet.shape[0]

for row in testSet.iterrows():
    row = row[1]
    if row['Label'] == row['predicted']:
        correct += 1
    else:
        if row['Label'] == "ham":
            falsePositive = falsePositive + 1
        elif row['Label'] == "spam":
            falseNegative = falseNegative + 1

print("Total number of messages in test dataset", testSet.shape[0])
print('Number of messages classified correctly:', correct)
print('Number of wrongly classified messages:', total - correct)
print("Falsenegative:", falseNegative)
print("Falsepositive:", falsePositive)
print('Accuracy achieved:', correct / total * 100)

# In[ ]:




