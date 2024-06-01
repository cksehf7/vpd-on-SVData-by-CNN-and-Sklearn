# vpd-on-SVData-by-CNN-and-Sklearn
*Term project for Advanced ML course   

A series of processes that perform VPD with SVD datasets.    
[Saarbrueken Voice Database](https://stimmdb.coli.uni-saarland.de/help_en.php4)    
We used only german pharse **"Guten morgen, wie geht es Ihnen".**, not susutained vowels(a, i, u), among the SVD datasets.   

The fixed training/test set has the following number of files.   
train healthy   : 532 wav files    
train_pathology : 762 wav files    
test_healthy    : 100 wav files     
test_pathology  : 100 wav files   

-----------------------------------------------------------------------------------------------------



This analysis utilized Sklearn and Pytorch and environment used Google Colab (GPU : T4).   
Also includes 2 Analazing methods.   

Analysis 1.   
We trained model by speech data with ScikitLearn based on feature extraction.   
The paper I'm referring to is [Link]      
Since I used the phase, the performance is not good and it comes out about 69%.   
The feature extraction uses .py,   
Use .ipynb for learning   
   
Analysis 2.   
Voice data is divided into spectrograms, melspectrograms, and MFCC   
(Detailed parameter settings are not covered in image creation.)   
For learning, we learned up to 20 efficientNet-b0 and averaged the prediction list to mimic the hard voting instance. The highest accuracy was 84.5%.   
(From personal experience, the efficientNet-b0 was chosen based on the judgment that it was the fastest and most accurate for this task among a single model.)   

Use the following .py to create an image file: make_spe, make_mel-spec, and make_mfcc      
Use ensemble.ipynb for learning   


