#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:33:19 2020

@author: r17935avinash
"""
from flair.data import Corpus
from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings
from flair.embeddings import FlairEmbeddings,StackedEmbeddings,TokenEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.datasets import CSVClassificationCorpus
from flair.visual.training_curves import Plotter
from flair.data import Sentence,Dictionary
import os
import sys
import torch

class classifier():
    def __init__(self,data_folder):
        self.column_name_map = {0: "text", 1: "label"}
        self.corpus: Corpus = CSVClassificationCorpus(data_folder,
                                         self.column_name_map,
                                         skip_header=True,
                                         delimiter='\t' ) 
        self.label_dict = self.corpus.make_label_dictionary()
        self.embeddings = [FlairEmbeddings('hi-forward'),FlairEmbeddings('hi-backward'),]
        self.download_path_split = torch.__file__.split("/")[:-2]
        self.download_dir = "/".join(self.download_path_split) + "/HindiNLP"
        self.dest_path = self.download_dir + "/resources/taggers/classifiers"

    def train(self,train_dict):

        if os.path.exists(self.dest_path) == False :
            if not os.path.exists(self.download_dir + "/resources"):
                os.mkdir(self.download_dir + "/resources")
            if not os.path.exists(self.download_dir + "/resources/tagger"):
                os.mkdir(self.download_dir + "/resources/tagger")
            if not os.path.exists(self.download_dir + "/resources/tagger/classifiers"):
                os.mkdir(self.download_dir + "/resources/tagger/classifiers")

        document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(self.embeddings,
                                                                     hidden_size=train_dict["hidden_size"],
                                                                     reproject_words=True,
                                                                     reproject_words_dimension=train_dict["output_size"])
        classifier = TextClassifier(document_embeddings, label_dictionary=self.label_dict)    
        trainer = ModelTrainer(classifier, self.corpus)
        trainer.train(self.dest_path,
              learning_rate=train_dict["lr"],
              mini_batch_size=train_dict["batch_size"],
              max_epochs=train_dict["n_epochs"])
        
    def Plot_Weights(self):
        if os.path.exists(dest_path) == False:
            print("Error, First Train your models")
            sys.exit()
        plotter = Plotter()
        plotter.plot_weights(self.dest_path+"/weights.txt")
        
    def Predict(input_text):
        if os.path.exists(dest_path) == False:
            print("Error, First Train your models")
            sys.exit()
        classifier = TextClassifier.load(self.dest_path+"/final-model.pt")
        sentence = Sentence(input_text)
        classifier.predict(sentence)
        return sentence.labels
    
    
#
# if __name__ == "__main__" :
#    SVC = classifier("../../Desktop/Hindi-NLI/BBC")
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
        
        
        
        
        
    
        