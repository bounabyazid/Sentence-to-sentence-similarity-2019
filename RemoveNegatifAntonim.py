from nltk.corpus import wordnet
class AntonymReplacer(object):
  def replace(self, word, pos=None):
    antonyms = set()
    for syn in wordnet.synsets(word, pos=pos):
      for lemma in syn.lemmas():
        for antonym in lemma.antonyms():
          antonyms.add(antonym.name())
    if len(antonyms) == 1:
       return antonyms.pop()
    else:
       return None

  def replace_negations(self, sent):
      i, l = 0, len(sent)
      words = []
      while i < l:
            word = sent[i]
            if word == 'not' and i+1 < l:
               ant = self.replace(sent[i+1])
               if ant:
                  words.append(ant)
                  i += 2
                  continue
            words.append(word)
            i += 1
      return words

#negative prefix
#_______________

#de dis in mis non un

#Negtive words
#_____________
    
#no not none no one nobody nothing nowhere never

#Negtive adverbs
#_______________

#Hardly scarcely barly
