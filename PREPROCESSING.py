import re
import nltk
import math
import scipy

import gensim
import pickle as pkl

from math import log
from scipy.stats import logistic
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as wn_ic

from nltk.corpus import sentiwordnet as wsn

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.tokenize import RegexpTokenizer

from RegExpReplacement import RegexpReplacer
from RemoveNegatifAntonim import AntonymReplacer

from string import punctuation
from nltk import word_tokenize
from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer

from matplotlib import pyplot as plt

from gensim.models import Word2Vec

def Extract_Numbers(string, ints=True):            
    numexp = re.compile(r'\d[\d,]*[\.]?[\d{2}]* ?')
    numbers = numexp.findall(string)
    numbers = [x.strip(' ') for x in numbers]

    return numbers

def Extract_Acronyms(sentence):
    Acronyms = re.findall('(?:(?<=\.|\s)[A-Z]\.)+',sentence)
    return Acronyms

def Remove_From_Sentence(sentence,List):
    for ele in List:
        sentence = sentence.replace(str(ele),'')
    return sentence

def Replace_Punctuations(sentence):
    sentence = sentence.replace('.',' ')
    sentence = sentence.replace('\'',' ')
    return sentence

def normalise(word):
    lemmatizer = nltk.WordNetLemmatizer()
    stemmer = nltk.stem.porter.PorterStemmer()

    word = word.lower()
    word = stemmer.stem_word(word)
    word = lemmatizer.lemmatize(word)
    return word

def normalise_words(words):
    Nwords = []
    for word in words:
        Nwords.append(normalise(word))
    return Nwords
  
def tokenize(text):
    stop_words = list(set(('a','also','an','and','as','at','but','by','for','from','in','it','its','of','on','or','that','the','to','may','is')))#+list(punctuation)
    stop_words.extend(list(set(stopwords.words('english'))))
    text = Replace_Punctuations(text)  
    words = word_tokenize(text)

    #words = normalise_words(words)
    
    return [w for w in words if w not in stop_words and not w.isdigit() and len(w)>1]

def PREPROCESSING(sentence):
    S = sentence
    S = S.lower()

    RegReplacer = RegexpReplacer()
    S = RegReplacer.replace(S)

    tokens = tokenize(S)

    return tokens

def CLEANING_FILE(Folder,inFile,k):

    DATABASE = open('DATASETS/'+Folder+'/'+inFile+'.txt','r')
    lines = DATABASE.readlines()
    DATABASE.close()

    CLEAN = open('CLEANED DATASETS/'+Folder+'/'+inFile+'.txt','w')

    DATASET = []

    print('DataSet '+Folder+' Cleaned '+inFile)

    i = 1
    for line in lines:
        if k == 0:
           parts = line.split('.')
            
           sentence1 = parts[0]
           sentence2 = parts[1]
        if k == 1:
           parts = line.split('\t')
            
           sentence1 = parts[3]
           sentence2 = parts[4]  
        if k == 2:
           parts = line.split('\t')
            
           sentence1 = parts[1]
           sentence2 = parts[2]
           
        T1 = PREPROCESSING(sentence1)
        T2 = PREPROCESSING(sentence2)

        Tokens1 = " ".join(T1)
        Tokens2 = " ".join(T2)

        DATASET.append(Tokens1)
        DATASET.append(Tokens2)

        print('_____________Clean '+str(i)+'______________')

        CLEAN.write(Tokens1+'\t'+Tokens2+'\n')
        i = i + 1

    CLEAN.close()

    return DATASET

def CLEANING_DATASET():
    In_House = CLEANING_FILE('In-house','sents',0)
    DATASETS = In_House
    
    OShea = CLEANING_FILE('O’Shea et al','sentsinorder',0)
    DATASETS.extend(OShea)

    MSRP = CLEANING_FILE('MSRP','msr_paraphrase_test',1)
    del MSRP[0]
    del MSRP[0]
    DATASETS.extend(MSRP)
    
    SICK = CLEANING_FILE('SICK','SICK',2)
    del SICK[0]
    del SICK[0]
    DATASETS.extend(SICK)

    vectorizer = CountVectorizer()
    vectorizer.fit_transform(DATASETS)
    
    print('Loading Google Pre-trained model ...')
    model = gensim.models.KeyedVectors.load_word2vec_format('/home/polo/Desktop/GoogleNews-vectors-negative300.bin', binary=True)
    print('Google Pre-trained model has been loaded')

    ZVec = np.zeros(300)

    #NVOC = #open('VOCCABULARY/WordNotInVoc.txt','w')

    VOCC = {}#open('VOCCABULARY/Voccabulary.txt','w')
    for w in vectorizer.vocabulary_:
        if w in model.wv.vocab:
           #VOCC.write(w+'\t'+' '.join(map(str,model[w]))+'\n')
           VOCC[w] = model[w]
        else:
           VOCC[w] = ZVec 
             #VOCC.write(w+'\t'+' '.join(map(str,ZVec))+'\n')
             #NVOC.write(w+'\n')
    #VOCC.close()
    #NVOC.close()
    Outfile = open('Vocabulary.pkl','wb')
    pkl.dump(VOCC,Outfile)
    Outfile.close()
    print('It is Done ...')
    return model

#________________________________________________________________________

def EMBEDDING_SIMILARITY(tokens1,tokens2):

    VOC = LOAD_VOCABULARY()

    n = len(tokens1)
    m = len(tokens2)

    sim1 = 0
    for word1 in tokens1:
        if word1 in tokens2:
           sim1 = sim1 + 1
        else:
            S1 = VOC[word1]
            maxsim = -1
            for word2 in tokens2:
                S2 = VOC[word2]
                cos = cosine_sim(S1,S2)    
                if cos > maxsim:
                   maxsim = cos
            if maxsim != -1:
               sim1 = sim1 + maxsim
    sim1 = round(sim1/n,2)

    print('Sim1(S1,S2) = '+str(sim1))

    sim2 = 0
    for word2 in tokens2:
        if word2 in tokens1:
           sim2 = sim2 + 1
        else:
            S2 = VOC[word2]
            maxsim = -1
            for word1 in tokens1:
                S1 = VOC[word1]
                cos = cosine_sim(S1,S2)    
                if cos > maxsim:
                   maxsim = cos
            if maxsim != -1:
               sim2 = sim2 + maxsim
    sim2 = round(sim2/m,2)

    print('Sim2(S2,S1) = '+str(sim2))

    sim = (sim1 + sim2)/2
    #sim = sim1 if m>=n else sim2
    #sim = sim1 if sim2>=sim1 else sim2
    
    return round(sim,2)

#________________________________________________________________________

def COMPUTE_SIMILARITY_FILE(inFile):
    
    CLEAN = open('Results/Clean_'+inFile+'.txt','r')
    lines = CLEAN.readlines()
    CLEAN.close()
    
    SIMS = open('Results/'+inFile+'_Similarities.txt','w')

    i = 1
    for line in lines:
        line = line.replace('\n','')
            
        parts = line.split('\t')

        T1 = parts[0].split(' ')
        T2 = parts[1].split(' ')

        Tokens1 = " ".join(T1)
        Tokens2 = " ".join(T2)

        print(Tokens1)
        print(Tokens2)
        
        sim = EMBEDDING_SIMILARITY(T1,T2)

        print('Similarity Min = '+str(sim))
        print('_____________'+str(i)+'______________')
        SIMS.write(str(sim)+'\n')

        i = i + 1

    SIMS.close()

#________________________________________________________________________

def DATASET_COMPUTE_SIMILARITY(k):
    COMPUTE_SIMILARITY_FILE('sentsinorder',k)
    #COMPUTE_SIMILARITY_FILE('sents',k)

#________________________________________________________________________

def Get_Tokens_Matrix(File):
    sentences = []

    CleanFile = open('Results/'+File+'.txt','r')
    lines = CleanFile.read().splitlines()

    for line in lines:
        parts = line.split('\t')
        
        S1 = parts[1].lower()
        sentences.append(S1.split(" "))
        
        S2 = parts[2].replace('\n','').lower()
        sentences.append(S2.split(" "))
    return sentences

#________________________________________________________________________

def VECTOR_AVRG(Tokens,VOC):
    vec = np.zeros(300)
    n = 0
    for word in Tokens:
        #if word in VOC.keys():
        vec += VOC[word]
        #if np.all(VOC[word] != 0):
        #   n+=1

    vec /= len(Tokens)
    return vec

def LOAD_VOCABULARY():
    VOCC = open('VOCCABULARY/Voccabulary.txt','r')
    lines = VOCC.readlines()

    VOC = {}
    
    for line in lines:
        parts = line.split('\t')
        word = parts[0]
        vec = [float(s) for s in parts[1].split(" ")]
        VOC[word] = vec

    return VOC


def COSINE_SIM_DATASET(model):
    VOC = LOAD_VOCABULARY()

    #print('Loading MSRPC Pre-trained model ...')
    #model = gensim.models.KeyedVectors.load_word2vec_format('Results/MSRPC.bin', binary=True)
    #print('MSRPC Pre-trained model has been loaded')

    COSINE_SIM_FILE('Clean_Train',VOC,model)

#________________________________________________________________________

def Word_Antonyms(word):
    antonyms = []
    for syn in wn.synsets(word):
        for l in syn.lemmas():
            if l.antonyms():
               antonyms.append(l.antonyms()[0].name())
    return antonyms



def w2vsim(sent1, sent2, w2vecModel):
    maxes = [max(x) for x in [[w2vecModel.similarity(s1, s2) for s2 in sent2] for s1 in sent1]]
    return np.mean(maxes)

#________________________________________________________________________

def Plot_Sims(MSims,HSims):
    fig, ax = plt.subplots()

    #plt.plot(MSims50, 'g', label = "machine Sents50")
    #plt.plot(MSims50, 'go')

    plt.plot(MSims, 'r', label = "machine Sim65 = "+str(VECTORS_CORRELATION(MSims,HSims,0)))
    plt.plot(MSims, 'ro')


    plt.plot(HSims, 'b', label = "Human Sim65")
    plt.plot(HSims, 'bo')

    plt.xlabel('Sentences Paire')
    plt.ylabel('Similarity')
    plt.title('Decressing Similarity')

    for i, txt in enumerate(MSims):
        ax.annotate('('+str(i)+', '+str(txt)+')', (i,MSims[i]))
    
    plt.grid()
    
    plt.legend(loc = 'upper right')
    #plt.legend(loc='lower right')
    plt.show()

#-------------DATASETS = CLEANING_DATASET()
  
#DATASET_COMPUTE_SIMILARITY(1)

##DATASET = CLEANING_FILE('sentsinorder',1)

#MSims = np.asarray([line.strip() for line in open('Results/sentsinorder_Similarities.txt', 'r')]).astype(float)
#MSims50 = np.asarray([line.strip() for line in open('Results/sents_Similarities.txt', 'r')]).astype(float)

#HSims = np.asarray([line.strip() for line in open('DATASET/sentsinordersims.txt', 'r')]).astype(float)
#HSims = np.divide(HSims, 4)

#Plot_Sims(MSims,HSims)
#print('La valeur de Correlation = '+str(VECTORS_CORRELATION(MSims,HSims,0)))

#print('MSims Standar Deviation = ',round(np.std(MSims),2),' mean =',round(np.mean(MSims),2))
#print('HSims Standar Deviation = ',round(np.std(HSims),2),' mean =',round(np.mean(HSims),2))

#print('MSims50 Standar Deviation = ',round(np.std(MSims50),2),' mean =',round(np.mean(MSims50),2))


