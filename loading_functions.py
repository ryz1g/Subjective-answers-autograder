import numpy as np
import time
from gensim.models.keyedvectors import KeyedVectors as ww
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize



t0=time.time()

#function to remove stop words,lemmatize and save the result in text files to be used for evaluation
def pre_process() :
    lemmatizer = WordNetLemmatizer()                #to lemmatize words
    stop_words = set(stopwords.words('english'))    #list of stop words in English

    #pre-processing of keywords starts here
    fh=open("keywords.txt")
    s1=fh.read().strip().split(",")
    for i in range(len(s1)) :
        s1[i]=s1[i].lower()
    fh=open("keywords.txt","w")
    for i in range(len(s1)-1) :
        w=s1[i]
        if w not in stop_words :
            lemtized=w
            for c in ["n", "a", "v", "r", "s"] :
                lemtized=lemmatizer.lemmatize(lemtized, pos=c)
            fh.write(lemtized)
            fh.write(",")
    fh.write(s1[len(s1)-1])
    #pre-processing of keywords ends and new keywords have been saved

    #pre-processing of key sentences starts here
    fh=open("keysen.txt")
    sentences=[]
    liist=[]
    for line in fh :
        sentences.append(line)
    for sent in sentences :
        s1=sent.strip().split()
        new_sent=""
        for i in range(len(s1)) :
            s1[i]=s1[i].lower()
        for i in range(len(s1)) :
            w=s1[i]
            if w not in stop_words :
                lemtized=w
                for c in ["n", "a", "v", "r", "s"] :
                    lemtized=lemmatizer.lemmatize(lemtized, pos=c)
                new_sent=new_sent+" "+lemtized
        liist.append(new_sent.lstrip())
    fh=open("keysen.txt", "w")
    for s in liist :
        fh.write(s)
        fh.write("\n")
    #pre-processing of key sentences ends here and new sentences have been saved

    #pre-processing of answer starts here
    fh=open("answer.txt")
    s=fh.read().strip().replace('\n',' ')
    li=sent_tokenize(s)
    liist=[]
    for sent in li :
        s1=sent.replace('.',' ').strip().split()
        new_sent=""
        for i in range(len(s1)) :
            s1[i]=s1[i].lower()
        for i in range(len(s1)) :
            w=s1[i]
            if w not in stop_words :
                lemtized=w
                for c in ["n", "a", "v", "r", "s"] :
                    lemtized=lemmatizer.lemmatize(lemtized, pos=c)
                new_sent=new_sent+" "+lemtized
        liist.append(new_sent.lstrip())
    se=""
    for s in liist :
        se=se+s
        se=se+'. '
    fh=open("answer.txt", "w")
    fh.write(se)
    #pre-processing of answer ends here and new answer is saved




#loading all word vectors in model
model = ww.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, limit=500000)

#returns a matrix after finding word vectors of the answer
def load_answer() :
    fh=open("answer.txt")
    s=fh.read().strip().replace('.',' ').split()
    ans=list()
    s1=list()
    for word in s:
        try :
            ans.append(model[word.lower()])                 #uses only lower case words
            s1.append(word.lower())
        except :
            dump=0
    return (s,s1,np.array(ans))          #return a list of words in answer and matrix of dim(n,300)

def load_answer_sent() :
    ans_sent_vec=[]
    fh=open("answer.txt")
    s=fh.read().split('.')
    s1=[]
    for line in s :
        s1.append(line.strip())
    for line in s1 :
        temp_s=line.split()
        bogus=0
        temp_v=np.squeeze(np.zeros((1,300)))
        for word in temp_s:
            try :
                temp_v=temp_v+model[word.lower()]
            except :
                bogus=bogus+1
        ans_sent_vec.append(temp_v/(len(temp_s)-bogus))
    return (s1,ans_sent_vec)



#returns a list of keywords and a list of word vectors of them taken from text file
def load_keywords() :
    fh=open("keywords.txt")
    s1=fh.read().strip().split(",")
    for i in range(len(s1)) :
        s1[i]=s1[i].lower()
    s2=list()
    s3=list()
    x=int(s1[len(s1)-1])
    c=0
    for word in s1:
        if c==(len(s1)-1) :
            break
        try :
            s2.append(model[word])
            s3.append(word)
        except :
            None
        c=c+1
    return (s1,s3,s2,x)

def load_keysent() :
    fh=open("keysen.txt")
    s1=[]
    for line in fh :
        s1.append(line.strip())
    return s1


#returns a list of compressed word vectors for each sentence
def load_keysent_vec() :
    fh=open("keysen.txt")
    s1=list()
    for line in fh :
        temp_s=line.strip().split()
        bogus=0
        temp_v=np.squeeze(np.zeros((1,300)))
        for word in temp_s:
            try :
                temp_v=temp_v+model[word.lower()]
            except :
                bogus=bogus+1
        s1.append(temp_v/(len(temp_s)-bogus))
    return s1


def load_all() :
    pre_process()
    x1=load_answer()
    x2=load_keywords()
    x3=load_answer_sent()
    return (x1[0], x1[1], x1[2], x3[0] ,x3[1], x2[0], x2[1], x2[2], x2[3], load_keysent(), load_keysent_vec())

print("Time to load model",time.time()-t0)  #approx 37 seconds to load the model
