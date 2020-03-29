#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 09:28:58 2020

@author: r17935avinash
"""
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import FlairEmbeddings,StackedEmbeddings,TokenEmbeddings
from typing import List
from flair.data import Sentence,Dictionary
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import flair
import torch
from flair.visual.training_curves import Plotter

class NER():
        def __init__(self):
            self.tag_type = "ner"
            self.embedding_types: List[TokenEmbeddings] = [FlairEmbeddings('hi-forward'),
                                                           FlairEmbeddings('hi-backward')]
            self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=self.embedding_types)
            
        def Predict(self,text,is_path=False,path=""):
           if is_path == False:
               path = "resources/tagger/example-ner/best-model.pt"
           sentence = Sentence(text)
           tagger = SequenceTagger.load(path)
           tagger.predict(sentence)
           return sentence.to_tagged_string()
       
       
        def Predict_textfile(self,textfile,is_path=False,path=""):
            if is_path == False:
                path = "resources/tagger/example-ner/best-model.pt"
            tagger = SequenceTagger.load(path)
            dest_path = textfile[:-4]+"__NER.txt"
            out_f = open(dest_path,"w")
            with open(textfile,"r") as f:
                for i,line in enumerate(f):
                    sentence = Sentence(line)
                    tagger.predict(sentence)
                    for word in sentence.to_dict(tag_type='ner')["entities"]:
                        out_f.write(word['text']+"\t"+word['type']+"\n")
                    out_f.write("\n")
                
        def train_NER(self,data_folder,train_file,dev_file,test_file,train_dict,is_gpu=False):
            columns = {0: 'text', 1: 'ner'}
            if is_gpu == True:
                flair.device = torch.device("cuda:0")
            corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file=train_file,
                              test_file=test_file,
                              dev_file=dev_file)
            
            tag_dictionary = corpus.make_tag_dictionary(tag_type=self.tag_type)
            tagger: SequenceTagger = SequenceTagger(hidden_size=train_dict["hidden_size"],
                                        embeddings=self.embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=self.tag_type,
                                        use_crf=True)
            trainer: ModelTrainer = ModelTrainer(tagger, corpus)
            trainer.train('resources/taggers/saved-models',
              learning_rate=train_dict["lr"],
              mini_batch_size=train_dict["batch_size"],
              max_epochs=train_dict["epochs"])
            
            
             
                
         #  self.tagger = 
           
#            
#if __name__ == "__main__" :
#    detect_ner = HindiNER()
##    sentence = detect_ner.Predict("अविनाश आगरा में रहता है")
##    print(sentence)
##    detect_ner.Predict_textfile("Sentences.txt")
#    train_dict = {
#            "lr" : 0.1 ,
#            "batch_size" : 32 ,
#            "epochs" : 150 ,
#            "hidden_size" : 256
#            }
#    detect_ner.train_NER(data_folder="/Users/r17935avinash/Desktop/Hindi-NLI/Ner",
#                         train_file="train.txt",
#                         dev_file="dev.txt",
#                         test_file="test.txt",
#                         train_dict=train_dict) 
#    
#    
#    
#    
#    
#    
            