
from nltk.tag.stanford import StanfordNERTagger
from nltk.tokenize import word_tokenize

def formatted_entities(classified_paragraphs_list):
    entities = {'persons': list(), 'organizations': list(), 'locations': list(), 'dates': list(), 'percent': list(), 'money': list(), 'time': list()}

    for classified_paragraph in classified_paragraphs_list:
        for entry in classified_paragraph:
            entry_value = entry[0]
            entry_type = entry[1]

            if entry_type == 'PERSON':
                entities['persons'].append(entry_value)

            elif entry_type == 'ORGANIZATION':
                entities['organizations'].append(entry_value)

            elif entry_type == 'LOCATION':
                entities['locations'].append(entry_value)

            elif entry_type == 'DATE':
                entities['dates'].append(entry_value)

            elif entry_type == 'PERCENT':
                entities['percent'].append(entry_value)

            elif entry_type == 'MONEY':
                entities['money'].append(entry_value)

            elif entry_type == 'TIME':
                entities['time'].append(entry_value)

    return entities
    
#________________________________________________________________________

def formatted_entities_Text(classified_Text):
    entities = {'persons': list(), 'organizations': list(), 'locations': list(), 'dates': list(), 'percent': list(), 'money': list(), 'time': list()}

    for entry in classified_Text:
        entry_value = entry[0]
        entry_type = entry[1]

        if entry_type == 'PERSON':
           entities['persons'].append(entry_value)

        elif entry_type == 'ORGANIZATION':
             entities['organizations'].append(entry_value)

        elif entry_type == 'LOCATION':
             entities['locations'].append(entry_value)

        elif entry_type == 'DATE':
             entities['dates'].append(entry_value)

        elif entry_type == 'PERCENT':
             entities['percent'].append(entry_value)

        elif entry_type == 'MONEY':
             entities['money'].append(entry_value)

        elif entry_type == 'TIME':
             entities['time'].append(entry_value)

    return entities
#________________________________________________________________________
    
path='stanford-ner-2017-06-09/'
tagger = StanfordNERTagger(path+'classifiers/english.muc.7class.distsim.crf.ser.gz',path+'stanford-ner.jar',encoding='utf-8')

#________________________________________________________________________

def NER_Dict(Sentence):#S1,S2):
    #T1 = word_tokenize(S1)
    #T2 = word_tokenize(S2)
    tokenized_paragraphs = list()
    tokenized_paragraphs.append(word_tokenize(Sentence))

    classified_paragraphs_list = tagger.tag_sents(tokenized_paragraphs)
    formatted_result = formatted_entities(classified_paragraphs_list)
    #print(formatted_result)
    return formatted_result

#________________________________________________________________________    

def Similarity_NER(Sentence1, Sentence2):
    N1 = NER_Dict(Sentence1)
    N2 = NER_Dict(Sentence2)

    #print(N1,'\n______________\n')
    #print(N2)
    SumMin = 0
        
    'Here we check if there is a NE of the same class or not' 
    for i in N1.keys():
        if len(N1[i]) > 0 and len(N2[i]) > 0:
           SumMin += min(len(N1[i]), len(N2[i]))
        else:
            if len(N1[i]) > 0:
               SumMin += len(N1[i])
            else:
                SumMin += len(N2[i])
              
    Card_N1 = sum(len(v) for k,v in N1.items())
    Card_N2 = sum(len(v) for k,v in N2.items())
    
    Sim1 = 0
    if Card_N1 != 0 and Card_N2 != 0:
       Sim1 = round(SumMin/(Card_N1+Card_N2),2)
    
    'Here we check the exact match of the same NE class'
    'Sim(ent1, ent2) = |ent1 ∩ ent2| / |ent1  ∪ ent2|'
    'similarity = len(ent1 & ent2) / len(ent1 | ent2)'
    
    SetA = set([value for values in N1.values() for value in values])
    SetB = set([value for values in N2.values() for value in values])
    
    Intersect_N1_N2 = SetA.intersection(SetB)
    Union_N1_N2 = SetA.intersection(SetB)
    
    Sim2 = 0
    if len(Union_N1_N2) > 0:
       Sim2 = round(len(Intersect_N1_N2)/len(Union_N1_N2),2)

    Sim = max(Sim1,Sim2)
    #print (SumMin,Card_N1,Card_N2,len(Intersect_N1_N2),len(Union_N1_N2),Sim)
    return Sim

#________________________________________________________________________
   
def SIMILARITY_FILE(Folder,inFile,k):

    DATABASE = open('DATASETS/'+Folder+'/'+inFile+'.txt','r')
    lines = DATABASE.readlines()
    DATABASE.close()

    SIMILARITIES = open('SIMILARITIES/'+Folder+'/NER_'+inFile+'.txt','w')

    print('DataSet '+Folder+' Similarities '+inFile)

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
           
        
        print('_____________Sim '+str(i)+'______________')

        SIMILARITIES.write(str(Similarity_NER(sentence1,sentence2))+'\n')
        i = i + 1

    SIMILARITIES.close()

def SIMILARITY_DATASETS():
    SIMILARITY_FILE('In-house','sents',0)
    SIMILARITY_FILE('O’Shea et al','sentsinorder',0)
    SIMILARITY_FILE('MSRP','msr_paraphrase_test',1)
    SIMILARITY_FILE('SICK','SICK',2)

#SIMILARITY_FILE('O’Shea et al','sentsinorder',0)
SIMILARITY_DATASETS()
s1 = 'John said that they will begin evaluating the exams next week.'
s2 = 'John said that test evaluation will be started by Wednesday.'

#Similarity_NER(s1, s2)

