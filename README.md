# Subjective-answers-autograder
This is an autograder for subjective answers which makes use of keywords and key senteces provided beforehand to evaluate the answer written. This uses neural networks, NLP and GoogleNews Pretrained word embeddings.

Download this file below and place it in the same folder as rest of the files of the repository.
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

Instruction for running the autograder:-
1) Download all the files and put them in the same folder.
2) Download the above mentioned file (around 1.5gb) and put in the same folder as rest of the files.
3) Run evaluate.py
   - The program will take less than 10 seconds to initialize before actual calculations start(This time is used to load the
     google file into primary memory)
   - Then the actual calculations start and it should not take more that a few milliseconds
   - After calculations are done, it will ask if you want to try again, at this point you can make changes in the three text
     files(answer.txt, keywords.txt and key sentences.txt) to run the evaluation again. This time and times after that, only
     a few milliseconds will be taken to run the program.
   - Again the program will run and you can run it as many times as you are making changes to the text files.
   
   
   
FORMAT FOR THE TEXT FILES:-

FOR answer.txt
  - This file contains the answer that is being checked.
  - There should be no punctuation marks in this file except full stop to denote the end of a sentence.
  - There should be spaces between words. 
  - Each sentence should be after the previous sentence, the sentences should not be in new lines.
  - Answer can be as long as you like.
  
FOR keysen.txt
  - This file contains the key sentences the examiner wants to check the answer for.
  - This should not contain any punctuations.
  - Each line should be a short sentence.
  - Do not club sentences.
  - There can be as many sentences as you like.
 
FOR keywords.txt
  - This file contains the keywords the examiner is looking for.
  - Keywords are a single word and are not groups of words.
  - Hyphenated words can be used but will only be matched if the student also uses the same hyphenatedd word. They should be 
    avoided if they can be avoided.
  - Each keyword should be separated by a comma without any spaces before or after the comma.
  - After the keywords, there should be a number denoting the pairs of synonyms present in the keywords.
    eg: CO2,H20,carbondioxide,water - The number should be 2
        Hello,Hi,Howdy - The number should be 3 as three pairs can be made (Hello,Hi)(Hello,Howdy)(Hi,Howdy)
  - There can be as many keywords as you like.

HOW TO ANALYZE RESULTS:-
  - For analysis there are three things, that the program returns.
  - Firstly, the percentage match (var:p_mat). This tells the determined matching and can take values (0.0,0.1,0.2,...,1.0)
    This can be directly multiplied with the maximum marks to get awarded marks. p_mat is computed by making some adjustments 
    to raw_match by calling function test6().
  - Second variable that is important in analyzing is the raw_match. It is the match that is predicted by the neural network
    after taking in grad_vec. This can give anomalous results hence test6() is applied to remove such behaviours.
  - The third variable is the grad_vec for each answer. This is a three dimensional vector. The first parameter is percentage
    of exact keywords present in the answer. (Higher is better, range is from 0.0 to 1.0). Second variable is used to compute
    the minimum distance of the each keyword from the answer words and then taking the average for each keyword. (Lower is 
    better, minimum is 0.0). Last parameter is the percentage similarity in the sentences of the answer and the key sentences
    provided.(Higher is better, range is 0.0 to 1.0)

For any other doubts related to the format, check the sample format already included in the respective files.
IMPORTANT:Since the program uses word embeddings trained on google news, American english spellings are recommended otherwise
evaluation may give strange results. Most importantly, the keywords and key sentences the examiner provides, they should be 
in American English.
