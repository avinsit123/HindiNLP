# HindiNLPTools

A specialized NLP library which provides tools to perform basic NLP tasks on Hindi Datasets. Currently, the library supports 
<ul>
  <li> <b> Named Entity Recognition </b>: The Library provides NER support to tag hindi sentences. Further you can apply NER to sentences in textfiles with running only one line of code. If you wish to train youe own NER model, you can do so without writing any script but in one line of code.</li>
  <li> <b> Part-Of-Speech Tagger </b> : </li>
  <li> <b> AutoClassifier </b> : Train your models on classification datasets just in one line of code. Finetune model using custom parameters</li>
<ul>
  
## NER Tagger
The NER Tagger identifies various parts of sentences and tags them with the type of entity they could represent. Currently the tags supported by our model are
<table> 
  <tr> <th>  Heading </th> <th> Tag Type  </th></tr>
  <tr> <td>  NEP </td> <td> Person  </td></tr>
  <tr> <td>  NED </td> <td> Designation  </td> </tr>
  <tr> <td>  NEO </td> <td> Object  </td> </tr>
  <tr> <td>  NEA </td> <td> Abbreviation  </td> </tr>
  <tr> <td>  NEB </td> <td> Brand  </td> </tr>
  <tr> <td>  NETP </td> <td> Title-Person  </td> </tr>
  <tr> <td>  NETO </td> <td> Title-Object </td> </tr>
  <tr> <td>  NEL </td> <td> Location </td> </tr>
  <tr> <td>  NETI </td> <td> Time </td> </tr>
  <tr> <td>  NEN </td> <td> Number </td> </tr>
  <tr> <td> NEM </td> <td> Measure </td> </tr>
  <tr> <td>  NETE </td> <td> Terms </td> </tr>
</table>

In order to use the Tagger, for one sentence 
```python 
from HindiNLPTools.HindiNer import NER
detect_ner = NER()
sentence = detect_ner.Predict("अविनाश आगरा में रहता है")
print(sentence)
 ```
 
 Print the sentence to see what the tagger found. Furthermore, entire textfiles can be processed and NER tags can be identified for all sentences
 ```python 
from HindiNLPTools.HindiNer import NER
detect_ner = NER()
detect_ner.Predict_textfile("/path/to/textfile.txt")
 ```
The input textfile must contain sentences one after each other in seperate lines. The annotated tags can be found in the file named 'textfile__NER.txt'.

### Train your own NER model
If you wish, to train your own NER model with specific NER tags you can do so in one line. First, create dictionary with training hyperparameters in the following format.
```python
    train_dict = {
            "lr" : 0.1 ,              # learning rate
            "batch_size" : 32 ,       # batch size
            "epochs" : 150 ,          # no.of epochs
            "hidden_size" : 256       # size of hidden_state LSTMs
            }
   ```
  Finally, pass it to the training code.
  ```python 
      detect_ner.train_NER(data_folder="/path/to/folder",  # folder path containing all test files
                         train_file="train.txt",            
                         dev_file="dev.txt",
                         test_file="test.txt",
                         train_dict=train_dict) 
  ```
  Further use the trained model to predict Sentences in tags
  ```python 
from HindiNLPTools.HindiNer import NER
detect_ner = NER()
sentence = detect_ner.Predict("अविनाश आगरा में रहता है",is_path=True,path="/path/to/trained/model")
print(sentence)
 ```
 
