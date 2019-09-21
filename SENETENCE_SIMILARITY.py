import re
import nltk
import sys

from scipy import spatial
import numpy as np
from numpy import arange

import math

import gensim

from PREPROCESSING import PREPROCESSING

import scipy.stats

import pickle as pkl

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix,precision_recall_fscore_support

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

def VECTORS_CORRELATION(tokens1,tokens2,model):

    WV = model.wv

    vec1 = np.zeros(300)
    vec2 = np.zeros(300)

    for word1 in tokens1:
        if word1 in WV.vocab:
           vec1 += WV[word1]
        else:
            #if word1 in tokens2:
            #   vec1 += WV['word']
            #else:
            if word1 not in tokens2:
                vec = vec1/len(tokens1)
                vec1 *= vec
                
    for word2 in tokens2:
        if word2 in WV.vocab:
           vec2 += WV[word2]
        else:
            #if word2 in tokens1:
            #   vec2 += WV['word']
            #else:
            if word2 not in tokens1:
                vec = vec2/len(tokens2)
                vec2 *= vec
    #return round(cosine_sim(vec1,vec2),2),0         
    return scipy.stats.pearsonr(vec1,vec2)[0],scipy.stats.pearsonr(vec1,vec2)[1]

def VECTORS_CORRELATION2(tokens1,tokens2,model):

    WV = model.wv
    
    wordsIn1 = []
    wordsIn2 = []

    for word1 in tokens1:
        if word1 in WV.vocab:
           wordsIn1.append(word1)
        
    for word2 in tokens2:
        if word2 in WV.vocab:
           wordsIn2.append(word2)

    #n = len(wordsIn1)
    #m = len(wordsIn2)
    
    n = len(tokens1)
    m = len(tokens2)
    
    Sim1 = 0
    for word1 in wordsIn1:
        MaxCorrel = 0
        for word2 in wordsIn2:
            Correl = round(scipy.stats.pearsonr(WV[word1],WV[word2])[0],2)
            if MaxCorrel < Correl:
               MaxCorrel = Correl
        Sim1 += MaxCorrel
        
    Sim1 /= n
    
    Sim2 = 0
    for word2 in wordsIn2:
        MaxCorrel = 0
        for word1 in wordsIn1:
            Correl = round(scipy.stats.pearsonr(WV[word1],WV[word2])[0],2)
            if MaxCorrel < Correl:
               MaxCorrel = Correl
        Sim2 += MaxCorrel
        
    Sim2 /= m
    
    Sim = Sim1 if n < m else Sim2
    return Sim
#________________________________________________________________________

def VECTORS_AVRG(tokens1,tokens2,model):

    WV = model.wv
    
    vec1 = np.zeros(300)
    vec2 = np.zeros(300)
    i = 0

    for word1 in tokens1:
        if word1 in WV.vocab:
           vec1 += WV[word1]
        else:
            if word1 in tokens2:
               vec1 += WV['word']
            else:
                vec = vec1/len(tokens2)
                vec1 *= vec
        i += 1
    vec1 /= len(tokens1)
    
    i = 0
    for word2 in tokens2:
        if word2 in WV.vocab:
           vec2 += WV[word2]
        else:
            if word2 in tokens1:
               vec2 += WV['word']
            else:
                vec = vec2/len(tokens2)
                vec2 *= vec
        i += 1
    vec2 /= len(tokens2)

    #return round(cosine_sim(vec1,vec2),2)
    return Euclidean_sim(vec1,vec2)

#________________________________________________________________________
    
def VECTORS_REPPORT07(tokens1,tokens2,model):

    WV = model.wv
    
    wordsIn1 = []
    wordsIn2 = []

    for word1 in tokens1:
        if word1 in WV.vocab:
           wordsIn1.append(word1)
        
    for word2 in tokens2:
        if word2 in WV.vocab:
           wordsIn2.append(word2)

    #n = len(tokens1)
    #m = len(tokens2)
    
    wordsIn1 = tokens1
    wordsIn2 = tokens2
    
    n = len(wordsIn1)
    m = len(wordsIn2)
     
    Sim1 = 0
    for word1 in wordsIn1:
        MaxSim = 0
        for word2 in wordsIn2:
            Sim = round(cosine_sim(WV[word1],WV[word2]),2)
            if MaxSim < Sim:
               MaxSim = Sim
        Sim1 += MaxSim
        
    Sim1 /= n
    
    Sim2 = 0
    for word2 in wordsIn2:
        MaxSim = 0
        for word1 in wordsIn1:
            Sim = round(cosine_sim(WV[word1],WV[word2]),2)
            if MaxSim < Sim:
               MaxSim = Sim
        Sim2 += MaxSim
       
    Sim2 /= m

    Sim = Sim2 if n < m else Sim1

    #Sim = (Sim1 + Sim2)/2

    return round(Sim,2)
    
#________________________________________________________________________

def OShea_SIMILARITY_EM():
    DATABASE = open('DATASETS/O’Shea et al/sentsinorder.txt','r')
    lines = DATABASE.readlines()
    DATABASE.close()

    print('Loading Google Pre-trained model ...')
    model = gensim.models.KeyedVectors.load_word2vec_format('/home/polo/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
    print('Google Pre-trained model has been loaded')
    
    #SIMILARITIES = open('SIMILARITIES/O’Shea et al/EM_sentsinorder.txt','w')
    
    SimsEM = []
    SimsAVG = []
    SimsR7 = []
    SimsCorrel = []
    i = 1
    for line in lines:
        parts = line.split('.')
        sentence1 = parts[0]
        sentence2 = parts[1]
        
        tokens1 = list(set(PREPROCESSING(sentence1)))
        tokens2 = list(set(PREPROCESSING(sentence2)))
        
        print('_____________'+str(i)+'______________')
        print(tokens1,tokens2)
                
        #SIMILARITIES.write(str(VECTORS_AVRG(tokens1,tokens2,model))+'\n')
        #SIMILARITIES.write(str(EMMBEDINGS_SIMLARITY(tokens1,tokens2,model))+'\n')
        SimsAVG.append(VECTORS_AVRG(tokens1,tokens2,model))
        SimsEM.append(EMMBEDINGS_SIMLARITY(tokens1,tokens2,model))
        SimsR7.append(VECTORS_REPPORT07(tokens1,tokens2,model))
        SimsCorrel.append(VECTORS_CORRELATION2(tokens1,tokens2,model))#[0])
        
        print('EM     = ',SimsEM[i-1])
        print('Sims AVG    = ',SimsAVG[i-1])
        print('Sims R7     = ',SimsR7[i-1])
        print('Sims Correl = ',SimsCorrel[i-1])

        i = i + 1

    #SIMILARITIES.close()
    
    Act_Sims = LOAD_ACTUAL_SIMILARITY('O’Shea et al','sentsinordersims',0)
        
    print(scipy.stats.pearsonr(Act_Sims,SimsEM)[0])
    print(scipy.stats.pearsonr(Act_Sims,SimsAVG)[0])
    print(scipy.stats.pearsonr(Act_Sims,SimsR7)[0])
    print(scipy.stats.pearsonr(Act_Sims,SimsCorrel)[0])
    
    Act_Sims  = list(np.divide(Act_Sims, 4))
    #Act_Sims = [ '%.2f' % elem for elem in Act_Sims ]


    print('________________ O’Shea et al ________________')
    
    Act_Sims = LOAD_ACTUAL_SIMILARITY_T('O’Shea et al','sentsinordersims',0,0.5)
    SimsCorrel = BinaryList(SimsCorrel,0.5)
    print(confusion_matrix(Act_Sims, SimsCorrel))

    accuracy = accuracy_score(Act_Sims, SimsCorrel)
    print('Correl Accuracy: %f' % accuracy)

    SimsR7 = BinaryList(SimsR7,0.5)
    print(confusion_matrix(Act_Sims, SimsR7))
    accuracy = accuracy_score(Act_Sims, SimsR7)
    print('R7 Accuracy: %f' % accuracy)
    
    #return SimsEM, SimsAVG, SimsR7, SimsCorrel, Act_Sims
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


def SIMILARITY_FILE(Folder,inFile,model):

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

        #SIMILARITIES.write(str(SIMILARITY_EM(sentence1,sentence2,Vocabulary))+'\n')
        
        tokens1 = PREPROCESSING(sentence1)
        tokens2 = PREPROCESSING(sentence2)
        
        #SIMILARITIES.write(str(VECTORS_AVRG(tokens1,tokens2,model))+'\n')
        SIMILARITIES.write(str(VECTORS_CORRELATION2(tokens1,tokens2,model))+'\n')
        i = i + 1

    SIMILARITIES.close()

def SIMILARITY_EM_DATASETS():
    print('Loading Google Pre-trained model ...')
    model = gensim.models.KeyedVectors.load_word2vec_format('/home/polo/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
    print('Google Pre-trained model has been loaded')

    SIMILARITY_FILE('In-house','sents',model)
    SIMILARITY_FILE('O’Shea et al','sentsinorder',model)
    SIMILARITY_FILE('MSRP','msr_paraphrase_test',model)
    SIMILARITY_FILE('SICK','SICK',model)

#________________________________________________________________________

def Similarity_AB(EM_Sim, Sim_NER,Alpha = 0.2):
    Sim = Alpha * EM_Sim + (1-Alpha) * Sim_NER
    return round(Sim,2)  
#________________________________________________________________________

def OVERALL_SIMILARITY_FILE(Folder,inFile):
    Sim_NER = []
    Sim_EM = []
    Dict = {}
   
    with open('SIMILARITIES/'+Folder+'/NER_'+inFile+'.txt','r') as f:
         for line in f:
             line = float(line)
             Sim_NER.append(line)
         
    with open('SIMILARITIES/'+Folder+'/EM_'+inFile+'.txt','r') as f:
         for line in f:
             line = float(line)
             Sim_EM.append(line)
    
    print('DataSet '+Folder+' Similarities '+inFile)
    for Alpha in np.arange (0.1, 0.9, 0.05):
        Sims = []
        for i in range (len(Sim_NER)):
            Sims.append(Similarity_AB(Sim_EM[i], Sim_NER[i], round(Alpha,2)))
        Dict[round(Alpha,2)] = Sims
    return Dict
    #with open('SIMILARITIES/'+Folder+'/OverAll_'+inFile+'.txt','w') as f: 
    #     for i in range (len(Sim_NER)):
    #         Sim = Similarity_AB(Sim_EM[i], Sim_NER[i])
    #         f.write(str(Sim)+'\n')
    #         print('_____________OverAll Sim '+str(i)+'______________')
    #f.close()
#________________________________________________________________________

def OVERALL_SIMILARITY_DATASETS():
    In_House = OVERALL_SIMILARITY_FILE('In-house','sents')
    OShea = OVERALL_SIMILARITY_FILE('O’Shea et al','sentsinorder')
    MSRP = OVERALL_SIMILARITY_FILE('MSRP','msr_paraphrase_test')
    SICK = OVERALL_SIMILARITY_FILE('SICK','SICK')
    
    return In_House, OShea, MSRP, SICK

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
        if k == 2:
           Act_Sims.append(round(float(line)/5.,2))
        i=+1
             
    return Act_Sims

#________________________________________________________________________

def BinaryList(List,T):
    Sims = []
    for elem in List:
        Sims.append(1 if float(elem) >= T else 0)
    return Sims

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

def Plotting():
    In_House, OShea, MSRP, SICK = OVERALL_SIMILARITY_DATASETS()
    
    Act_Sims = list(range(50,0,-1))
    In_House_Correl = []
    for Alpha in OShea.keys():
        In_House_Correl.append((round(scipy.stats.pearsonr(Act_Sims, In_House[Alpha])[0],2)))
   
    Act_Sims = LOAD_ACTUAL_SIMILARITY('O’Shea et al','sentsinordersims',0)
    OShea_Correl = []    
    for Alpha in OShea.keys():
        OShea_Correl.append((round(scipy.stats.pearsonr(Act_Sims, OShea[Alpha])[0],2)))

    Act_Sims = LOAD_ACTUAL_SIMILARITY('MSRP','msr_paraphrase_test',1)   
    MSRP_Correl = []     
    for Alpha in OShea.keys():
        MSRP_Correl.append((round(scipy.stats.pearsonr(Act_Sims, MSRP[Alpha])[0],2)))

    Act_Sims = LOAD_ACTUAL_SIMILARITY('SICK','SICK',2)    
    SICK_Correl = []     
    for Alpha in OShea.keys():
        SICK_Correl.append((round(scipy.stats.pearsonr(Act_Sims, SICK[Alpha])[0],2)))

    x_val = [*OShea]#OShea.keys()
    
    plt.plot(x_val,In_House_Correl,'*c',label='In House')
    plt.plot(x_val,OShea_Correl,'+g',label='OShea')
    plt.plot(x_val,MSRP_Correl,'xb',label='MSRP')
    plt.plot(x_val,SICK_Correl,'or',label='SICK')

    plt.xlabel('Alpha')
    plt.ylabel('Pearson Coefficient')
    #plt.title("Simple Plot")
    
    plt.legend()
    plt.savefig('Alpha_vs_Pearson_Coefficient.eps', format='eps')

    plt.show()

    
    print('__________________ IN HOUSE __________________')
    Act_Sims = BinaryList(['%.2f' % elem for elem in list(arange(1.,0.,-0.02))],0.7)
    for Alpha in OShea.keys():
        Y = [ '%.2f' % elem for elem in In_House[Alpha]]
        Y = BinaryList(Y,0.7)
        print('_____Alpha = '+str(Alpha)+'_____\n')
        precision = precision_score(Act_Sims, Y)
        print('Precision: %.2f' % precision)
        recall = recall_score(Act_Sims, Y)
        print('Recall: %.2f' % recall)
  
    print('________________ O’Shea et al ________________')
    Act_Sims = LOAD_ACTUAL_SIMILARITY_T('O’Shea et al','sentsinordersims',0,0.7)
    for Alpha in OShea.keys():
        Y = ['%.2f' % elem for elem in OShea[Alpha]]
        Y = BinaryList(Y,0.7)
        print('_____Alpha = '+str(Alpha)+'_____\n')
        precision = precision_score(Act_Sims, Y)
        print('Precision: %.2f' % precision)
        recall = recall_score(Act_Sims, Y)
        print('Recall: %.2f' % recall)
        accuracy = accuracy_score(Act_Sims, Y)
        print('Accuracy: %f' % accuracy)
        
    print('____________________ MSRP ____________________')
    Act_Sims = LOAD_ACTUAL_SIMILARITY_T('MSRP','msr_paraphrase_test',1,0.6)   
    for Alpha in OShea.keys():
        Y = [ '%.2f' % elem for elem in MSRP[Alpha]]
        Y = BinaryList(Y,0.5)
        print('_____Alpha = '+str(Alpha)+'_____\n')
        precision = precision_score(Act_Sims, Y)
        print('Precision: %.2f' % precision)
        recall = recall_score(Act_Sims, Y)
        print('Recall: %.2f' % recall)
        accuracy = accuracy_score(Act_Sims, Y)
        print('Accuracy: %f' % accuracy)
        
    print('____________________ SICK ____________________')
    Act_Sims = LOAD_ACTUAL_SIMILARITY_T('SICK','SICK',2,0.85)   
    for Alpha in OShea.keys():
        Y = [ '%.2f' % elem for elem in SICK[Alpha]]
        Y = BinaryList(Y,0.85)
        print('_____Alpha = '+str(Alpha)+'_____\n')
        precision = precision_score(Act_Sims, Y)
        print('Precision: %.2f' % precision)
        recall = recall_score(Act_Sims, Y)
        print('Recall: %.2f' % recall)
        accuracy = accuracy_score(Act_Sims, Y)
        print('Accuracy: %f' % accuracy)
    #Y = [ '%.2f' % elem for elem in In_House[0.85] ]
    #print (Y)
    
    # accuracy: (tp + tn) / (p + n)
    #accuracy = accuracy_score(Act_Sims, Y)
    #print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    #precision = precision_score(Act_Sims, Y)
    #print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    #recall = recall_score(Act_Sims, Y)
    #print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    #f1 = f1_score(Act_Sims, Y)
    #print('F1 score: %f' % f1)
    
    return In_House_Correl, OShea_Correl, MSRP_Correl, SICK_Correl

def Plotting_OShea():
    OShea = []
    Sim_NER = []
    Sim_EM = []
    
    with open('SIMILARITIES/O’Shea et al/NER_sentsinorder.txt','r') as f:
         for line in f:
             line = float(line)
             Sim_NER.append(line)
         
    with open('SIMILARITIES/O’Shea et al/EM_sentsinorder.txt','r') as f:
         for line in f:
             line = float(line)
             Sim_EM.append(line)
    
    for i in range (len(Sim_NER)):
        OShea.append(Similarity_AB(Sim_EM[i], Sim_NER[i], 0.8))
        
    Act_Sims = LOAD_ACTUAL_SIMILARITY('O’Shea et al','sentsinordersims',2)
    
    
    x_val = list(range(1, 66))
    
    plt.plot(x_val,OShea,'-ro',label='Our Formula')
    plt.plot(x_val,Act_Sims,'-bo',label='OShea')
    
    plt.xlabel('Sentences Paire')
    plt.ylabel('Similarity')
    #plt.title("Simple Plot")
    
    plt.legend()
    plt.savefig('O’Shea_et_al.eps', format='eps')

    plt.show()
    
    return Act_Sims, OShea

def Measure_Diff_Sim(x):
    return round(sum([math.pow(x[n]-x[n-1], 2) for n in range(1,len(x))]), 2)

def Evaluate_In_House():
    
    Sim_NER = []
    Sim_EM = []
    
    Wup = []
    Yago_Gaph = []
    Yago_Paph = []
    DBPedia = []
    W2V = []

    Alpha_IN_HOUSE = {}
    Diff_IN_HOUSE = {}
    with open('/home/polo/.config/spyder-py3/Paper-For-Oulu-PhD-master/SIMILARITIES/In-house/NER_sents.txt','r') as f:
         for line in f:
             Sim_NER.append(float(line))
         
    with open('/home/polo/.config/spyder-py3/Paper-For-Oulu-PhD-master/SIMILARITIES/In-house/EM_sents.txt','r') as f:
         for line in f:
             Sim_EM.append(float(line))
    for Alpha in np.arange (0.1, 0.9, 0.05):
        IN_HOUSE = []
        for i in range (len(Sim_NER)):
            IN_HOUSE.append(Similarity_AB(Sim_EM[i], Sim_NER[i], round(Alpha,2)))
        Alpha_IN_HOUSE[round(Alpha,2)] = IN_HOUSE
        Diff_IN_HOUSE[round(Alpha,2)] = Measure_Diff_Sim(IN_HOUSE)
        
    with open('/home/polo/.config/spyder-py3/Paper-For-Oulu-PhD-master/nlpprojectmeetingforinstallation/Results/wup_sims.txt','r') as f:
         for line in f:
             Wup.append(float(line))
             
    with open('/home/polo/.config/spyder-py3/Paper-For-Oulu-PhD-master/nlpprojectmeetingforinstallation/Results/yago_graph.txt','r') as f:
         for line in f:
             Yago_Gaph.append(float(line))
             
    with open('/home/polo/.config/spyder-py3/Paper-For-Oulu-PhD-master/nlpprojectmeetingforinstallation/Results/yago_path.txt','r') as f:
         for line in f:
             Yago_Paph.append(float(line))
             
    with open('/home/polo/.config/spyder-py3/Paper-For-Oulu-PhD-master/nlpprojectmeetingforinstallation/Results/dbpedia.txt','r') as f:
         for line in f:
             DBPedia.append(float(line))
    
    with open('/home/polo/.config/spyder-py3/Paper-For-Oulu-PhD-master/nlpprojectmeetingforinstallation/Results/Google_w2vec_sim.txt','r') as f:
         for line in f:
             W2V.append(float(line))
         
    Diff_Wup = Measure_Diff_Sim(Wup)
    Diff_Yago_Gaph = Measure_Diff_Sim(Yago_Gaph)
    Diff_Yago_Paph = Measure_Diff_Sim(Yago_Paph)
    Diff_DBPedia = Measure_Diff_Sim(DBPedia)
    Diff_W2V = Measure_Diff_Sim(W2V)
    
    return Diff_Wup, Diff_Yago_Gaph, Diff_Yago_Paph, Diff_DBPedia, Diff_W2V, Diff_IN_HOUSE

def Plotting_In_House(Alpha):
    IN_HOUSE = []
    Sim_NER = []
    Sim_EM = []
    
    Wup = []
    Yago_Gaph = []
    Yago_Paph = []
    DBPedia = []
    W2V = []
    
    with open('/home/polo/.config/spyder-py3/Paper-For-Oulu-PhD-master/SIMILARITIES/In-house/NER_sents.txt','r') as f:
         for line in f:
             Sim_NER.append(float(line))
         
    with open('/home/polo/.config/spyder-py3/Paper-For-Oulu-PhD-master/SIMILARITIES/In-house/EM_sents.txt','r') as f:
         for line in f:
             Sim_EM.append(float(line))
    
    for i in range (len(Sim_NER)):
        IN_HOUSE.append(Similarity_AB(Sim_EM[i], Sim_NER[i], Alpha))
        
    with open('/home/polo/.config/spyder-py3/Paper-For-Oulu-PhD-master/nlpprojectmeetingforinstallation/Results/wup_sims.txt','r') as f:
         for line in f:
             Wup.append(float(line))
             
    with open('/home/polo/.config/spyder-py3/Paper-For-Oulu-PhD-master/nlpprojectmeetingforinstallation/Results/yago_graph.txt','r') as f:
         for line in f:
             Yago_Gaph.append(float(line))
             
    with open('/home/polo/.config/spyder-py3/Paper-For-Oulu-PhD-master/nlpprojectmeetingforinstallation/Results/yago_path.txt','r') as f:
         for line in f:
             Yago_Paph.append(float(line))
             
    with open('/home/polo/.config/spyder-py3/Paper-For-Oulu-PhD-master/nlpprojectmeetingforinstallation/Results/dbpedia.txt','r') as f:
         for line in f:
             DBPedia.append(float(line))
    
    with open('/home/polo/.config/spyder-py3/Paper-For-Oulu-PhD-master/nlpprojectmeetingforinstallation/Results/Google_w2vec_sim.txt','r') as f:
         for line in f:
             W2V.append(float(line))
             
    Diff_Wup, Diff_Yago_Gaph, Diff_Yago_Paph, Diff_DBPedia, Diff_W2V, Diff_IN_HOUSE = Evaluate_In_House()
     
    x_val = list(range(1, 51))
    
    plt.plot(x_val,Wup,'-co',label='Diff Wup = '+str(Diff_Wup))
    plt.plot(x_val,W2V,'-ko',label='Diff W2V = '+str(Diff_Wup))
    plt.plot(x_val,DBPedia,'-go',label='Diff DBPedia = '+str(Measure_Diff_Sim(DBPedia)))

    plt.plot(x_val,Yago_Gaph,'-bo',label='Diff Yago Gaph = '+str(Diff_Yago_Gaph))
    plt.plot(x_val,Yago_Paph,'-mo',label='Diff Yago Paph = '+str(Diff_Yago_Paph))

    plt.plot(x_val,IN_HOUSE,'-ro',label='Our Formula Alpha = '+str(Alpha)+',\n Diff = '+str(Diff_IN_HOUSE[round(Alpha,2)]))

    
    plt.xlabel('Sentences Paire')
    plt.ylabel('Similarity')
    #plt.title("Simple Plot")
    
    plt.legend(bbox_to_anchor=(0, -0.15, 1, 0), loc=2, ncol=2, mode="expand", borderaxespad=0)
    
    #plt.legend(loc='best')
    plt.savefig('IN_HOUSE_FIGURES/In_House_'+str(Alpha)+'.eps', format='eps',
                 bbox_inches='tight')

    plt.show()
    
    return Measure_Diff_Sim(IN_HOUSE)

for Alpha in np.arange (0.1, 0.9, 0.05):
    Diff = Plotting_In_House(Alpha)


#------SIMILARITY_EM_DATASETS()
#OVERALL_SIMILARITY_FILE('In-house','sents')
#OVERALL_SIMILARITY_FILE('O’Shea et al','sentsinorder')

#OVERALL_SIMILARITY_DATASETS()

#EVALUATION(1)

#In_House_Correl, OShea_Correl, MSRP_Correl, SICK_Correl = Plotting()

#In_House_Correl = [ '%.2f' % elem for elem in In_House_Correl ]
#OShea_Correl = [ '%.2f' % elem for elem in OShea_Correl ]
#MSRP_Correl = [ '%.2f' % elem for elem in MSRP_Correl ]
#SICK_Correl = [ '%.2f' % elem for elem in SICK_Correl ]

#OShea_SIMILARITY_EM()

Act_Sims, OShea = Plotting_OShea()
