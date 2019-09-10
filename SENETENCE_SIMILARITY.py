import re
import nltk
import sys

from scipy import spatial
import numpy as np
import math

import gensim

from PREPROCESSING import *

import scipy.stats

import pickle as pkl

from sklearn.metrics import confusion_matrix

def Euclidean_sim(a,b):
    return scipy.spatial.distance.euclidean(a,b)

def cosine_sim(a, b):
    return scipy.spatial.distance.cosine(a, b)
    #return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def VECTORS_CORRELATION(vec1,vec2,i):
    return round(scipy.stats.pearsonr(vec1,vec2)[i],2)

#________________________________________________________________________

def EMMBEDINGS_SIMLARITY(tokens1,tokens2,model):

    WV = model.wv

    wordsIn1 = []
    wordsIn2 = []

    for word1 in tokens1:
        if word1 in WV.vocab:
           wordsIn1.append(word1)
        
    for word2 in tokens2:
        if word2 in WV.vocab:
           wordsIn2.append(word2)

    n = len(wordsIn1)
    m = len(wordsIn2)

    HamDiff = np.zeros((n, m))
    MaxSim = np.zeros(m)
    MaxSim2 = np.zeros(n)

    for j in range(m-1):
        vec2 = model[wordsIn2[j]]
        for i in range(n-1):
            vec1 = model[wordsIn1[i]]
            for k in range(299):
                if vec1[k] == vec2[k]:
                   HamDiff[i][j] = HamDiff[i][j]+1
            HamDiff[i][j] /= 300
            if HamDiff[i][j] > MaxSim[j]:
               MaxSim[j] = HamDiff[i][j]
    for i in range(n-1):
        for j in range(m-1):
            MaxSim2[i] =  np.max(HamDiff[i,:])/m
            
    #sim = sum(MaxSim)/m
    #le cas de division sur m ma Tansahech
    sim = ( (sum(MaxSim)/len(tokens2)) + (sum(MaxSim2)/len(tokens1)) )/2

    return round(sim,2)
#________________________________________________________________________

def VECTORS_CORRELATION(sentence1,sentence2,model):

    WV = model.wv

    tokens1 = PREPROCESSING(sentence1)
    tokens2 = PREPROCESSING(sentence2)

    vec1 = np.zeros(300)
    vec2 = np.zeros(300)

    for word1 in tokens1:
        if word1 in WV.vocab:
           vec1 += WV[word1]

    for word2 in tokens2:
        if word2 in WV.vocab:
           vec2 += WV[word2]

    return scipy.stats.pearsonr(vec1,vec2)[0],scipy.stats.pearsonr(vec1,vec2)[1]

#________________________________________________________________________

def VECTORS_AVRG(tokens1,tokens2,model):

    WV = model.wv
    
    vec1 = np.zeros(300)
    vec2 = np.zeros(300)
    i = 0

    for word1 in tokens1:
        if word1 in WV.vocab:
           vec1 += WV[word1]
        i += 1
    vec1 /= len(tokens1)
    
    i = 0
    for word2 in tokens2:
        if word2 in WV.vocab:
           vec2 += WV[word2]
        i += 1
    vec2 /= len(tokens2)

    #return cosine_sim(vec1,vec2)
    return Euclidean_sim(vec1,vec2)

#________________________________________________________________________

#________________________________________________________________________
                

def LOAD_VOCCABULARY():
    Infile = open('Vocabulary.pkl','rb')
    Vocabulary = pkl.load(Infile)
    
    return Vocabulary
#________________________________________________________________________

def SIMILARITY_EM(sentence1,sentence2,Vocabulary):
        
    tokens1 = PREPROCESSING(sentence1)
    tokens2 = PREPROCESSING(sentence2)

    vec1 = np.zeros(300)
    vec2 = np.zeros(300)

    for word1 in tokens1:
        if word1 in Vocabulary.keys():
           vec1 += Vocabulary[word1]

    for word2 in tokens2:
        if word2 in Vocabulary.keys():
           vec2 += Vocabulary[word2]
    
    return round(cosine_sim(vec1, vec2),2)      
#________________________________________________________________________


def SIMILARITY_FILE(Folder,inFile):

    DATABASE = open('CLEANED DATASETS/'+Folder+'/'+inFile+'.txt','r')
    lines = DATABASE.readlines()
    DATABASE.close()

    SIMILARITIES = open('SIMILARITIES/'+Folder+'/EM_'+inFile+'.txt','w')

    print('DataSet '+Folder+' Similarities '+inFile)
    
    Vocabulary = LOAD_VOCCABULARY()

    i = 1
    for line in lines:
        parts = line.split('\t')

        sentence1 = parts[0]
        sentence2 = parts[1]
        
        print('_____________Sim '+str(i)+'______________')

        SIMILARITIES.write(str(SIMILARITY_EM(sentence1,sentence2,Vocabulary))+'\n')
        i = i + 1

    SIMILARITIES.close()

def SIMILARITY_EM_DATASETS():
    SIMILARITY_FILE('In-house','sents')
    SIMILARITY_FILE('O’Shea et al','sentsinorder')
    SIMILARITY_FILE('MSRP','msr_paraphrase_test')
    SIMILARITY_FILE('SICK','SICK')

#________________________________________________________________________

def Similarity_AB(EM_Sim, Sim_NER):
    Alpha = 0.2
    Sim = Alpha * EM_Sim + (1-Alpha) * Sim_NER
    return round(Sim,2)        
#________________________________________________________________________

def OVERALL_SIMILARITY_FILE(Folder,inFile):
    Sim_NER = []
    Sim_EM = []
    
    with open('SIMILARITIES/'+Folder+'/NER_'+inFile+'.txt','r') as f:
         for line in f:
             line = float(line)
             Sim_NER.append(line)
         
    with open('SIMILARITIES/'+Folder+'/EM_'+inFile+'.txt','r') as f:
         for line in f:
             line = float(line)
             Sim_EM.append(line)
    
    print('DataSet '+Folder+' Similarities '+inFile)    

    with open('SIMILARITIES/'+Folder+'/OverAll_'+inFile+'.txt','w') as f: 
         for i in range (len(Sim_NER)):
             Sim = Similarity_AB(Sim_EM[i], Sim_NER[i])
             f.write(str(Sim)+'\n')
             print('_____________OverAll Sim '+str(i)+'______________')
    f.close()
#________________________________________________________________________

def OVERALL_SIMILARITY_DATASETS():
    OVERALL_SIMILARITY_FILE('In-house','sents')
    OVERALL_SIMILARITY_FILE('O’Shea et al','sentsinorder')
    OVERALL_SIMILARITY_FILE('MSRP','msr_paraphrase_test')
    OVERALL_SIMILARITY_FILE('SICK','SICK')
    
#________________________________________________________________________

def LOAD_SIMILARITY_T(Folder,inFile,T):
    Sims = []    
    with open('SIMILARITIES/'+Folder+'/OverAll_'+inFile+'.txt','r') as f:
         lines = f.readlines()
         for line in lines:
             Sims.append(1 if float(line) >= T else 0)
    return Sims
    
def LOAD_ACTUAL_SIMILARITY_T(Folder,inFile,k,T):
    DATABASE = open('DATASETS/'+Folder+'/'+inFile+'.txt','r')
    lines = DATABASE.readlines()
    DATABASE.close()
    
    Act_Sims = []
    i = 0
    for line in lines:
        if k == 0:
           Sim = round(float(line)/4.,2)
           Act_Sims.append(1 if Sim >= T else 0)
        if k == 1 and i > 0:
           Sim = float(line.split('\t')[0])
           Act_Sims.append(1 if Sim >= T else 0)
        if k == 2 and i > 0:
           Sim = round(float(line.split('\t')[4])/5.,2) 
           Act_Sims.append(1 if Sim >= T else 0)
        i=+1
             
    return Act_Sims
#________________________________________________________________________

def LOAD_SIMILARITY(Folder,inFile):
    Sims = []    
    with open('SIMILARITIES/'+Folder+'/OverAll_'+inFile+'.txt','r') as f:
         lines = f.readlines()
         for line in lines:
             Sims.append(float(line))
    return Sims
    
def LOAD_ACTUAL_SIMILARITY(Folder,inFile,k):
    DATABASE = open('DATASETS/'+Folder+'/'+inFile+'.txt','r')
    lines = DATABASE.readlines()
    DATABASE.close()
    
    Act_Sims = []
    i = 0
    for line in lines:
        if k == 0:
           Act_Sims.append(float(line))
        if k == 1 and i > 0:
           Act_Sims.append(float(line.split('\t')[0]))
        if k == 2 and i > 0:
           Act_Sims.append(float(line.split('\t')[4]))
        i=+1
             
    return Act_Sims

#________________________________________________________________________

def EVALUATION(k):
    if k == 0:     
       print('________________ O’Shea et al ________________')
    
       Act_Sims = LOAD_ACTUAL_SIMILARITY_T('O’Shea et al','sentsinordersims',0,0.67)
       Sims = LOAD_SIMILARITY_T('O’Shea et al','sentsinorder',0.7)
       print(confusion_matrix(Act_Sims, Sims))
    
       print('____________________ MSRP ____________________')
    
       Act_Sims = LOAD_ACTUAL_SIMILARITY_T('MSRP','msr_paraphrase_test',1,0.65)
       Sims = LOAD_SIMILARITY_T('MSRP','msr_paraphrase_test',0.7)
       print(confusion_matrix(Act_Sims, Sims))
    
       print('____________________ SICK ____________________')
    
       Act_Sims = LOAD_ACTUAL_SIMILARITY_T('SICK','SICK',2,0.85)
       Sims = LOAD_SIMILARITY_T('SICK','SICK',0.7)
       print(confusion_matrix(Act_Sims, Sims))
       print('______________________________________________')
    if k == 1:       
       print('________________ O’Shea et al ________________')
    
       Act_Sims = LOAD_ACTUAL_SIMILARITY('O’Shea et al','sentsinordersims',0)
       Sims = LOAD_SIMILARITY('O’Shea et al','sentsinorder')
       #print(round(scipy.stats.pearsonr(np.array(Act_Sims), np.array(Sims),0),2))
       print(abs(round(scipy.stats.pearsonr(Act_Sims, Sims)[0],2)))

       print('____________________ MSRP ____________________')
    
       #Act_Sims = LOAD_ACTUAL_SIMILARITY('MSRP','msr_paraphrase_test',1)
       #Sims = LOAD_SIMILARITY('MSRP','msr_paraphrase_test')
       #print(confusion_matrix(Act_Sims, Sims))
    
       print('____________________ SICK ____________________')
    
       #Act_Sims = LOAD_ACTUAL_SIMILARITY('SICK','SICK',2)
       #Sims = LOAD_SIMILARITY('SICK','SICK',0.7)
       #print(confusion_matrix(Act_Sims, Sims))
       print('______________________________________________')
    
#------SIMILARITY_EM_DATASETS()
#OVERALL_SIMILARITY_FILE('In-house','sents')
#OVERALL_SIMILARITY_FILE('O’Shea et al','sentsinorder')

#OVERALL_SIMILARITY_DATASETS()

EVALUATION(1)

#Act_Sims = LOAD_ACTUAL_SIMILARITY('O’Shea et al','sentsinordersims',0,0.67)
#Sims = LOAD_SIMILARITY('O’Shea et al','sentsinorder',0.7)
#print(confusion_matrix(Act_Sims, Sims))
