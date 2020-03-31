#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:33:19 2020

@author: r17935avinash
"""
from flair.data import Corpus
from flair.datasets import TREC_6
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings
from flair.embeddings import FlairEmbeddings,StackedEmbeddings,TokenEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.datasets import CSVClassificationCorpus
from typing import List
from flair.visual.training_curves import Plotter
from flair.data import Sentence,Dictionary
import os

class classifier():
    def __init__(self,data_folder):
        self.column_name_map = {0: "text", 1: "label"}
        self.corpus: Corpus = CSVClassificationCorpus(data_folder,
                                         self.column_name_map,
                                         skip_header=True,
                                         delimiter='\t' ) 
        self.label_dict = self.corpus.make_label_dictionary()
        self.embeddings = [FlairEmbeddings('hi-forward'),FlairEmbeddings('hi-backward'),]

    def train(self,train_dict,dest_path= os.getcwd() + '/HindiNLPTools/resources/taggers/classifiers'):
        document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(self.embeddings,
                                                                     hidden_size=train_dict["hidden_size"],
                                                                     reproject_words=True,
                                                                     reproject_words_dimension=train_dict["output_size"])
        classifier = TextClassifier(document_embeddings, label_dictionary=self.label_dict)    
        trainer = ModelTrainer(classifier, self.corpus)
        trainer.train(dest_path,
              learning_rate=train_dict["lr"],
              mini_batch_size=train_dict["batch_size"],
              max_epochs=train_dict["n_epochs"])
        
    def Plot_Weights(self,dest_path=os.getcwd() + '/HindiNLPTools/resources/taggers/classifiers'):
        plotter = Plotter()
        plotter.plot_weights(dest_path+"/weights.txt")
        
    def Predict(input_text,dest_path= os.getcwd() + '/HindiNLPTools/resources/taggers/classifiers'):
        classifier = TextClassifier.load(dest_path+"/final-model.pt")
        sentence = Sentence(input_text)
        classifier.predict(sentence)
        return sentence.labels
    
    

#if __name__ == "__main__" :
#    SVC = classifier("../Hindi-NLI/BBC")
#    train_dict = {
#            "hidden_size" : 512,
#            "output_size" : 256,
#            "lr" : 0.1 ,
#            "batch_size" : 256 ,
#            "n_epochs" : 150
#            }
#    SVC.train(train_dict)
#    SVC.predict("I am a good man")
#    
#        
#        
#
#        
#        
#        
#        
#                                                      
#        
        
        

        
        
        
        # tab-separated files) 
        
        
        
        
        
    
        