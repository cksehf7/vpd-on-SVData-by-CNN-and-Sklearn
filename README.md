# vpd-on-SVData-by-CNN-and-Sklearn
*Codes from my term project for Advanced ML course*   

This project contains a series of processes to do VPD(Voice Pathology Detection) task with SVD datasets.
[Saarbrueken Voice Database](https://stimmdb.coli.uni-saarland.de/help_en.php4)    
We used only the German phrase **"Guten Morgen, Wie geht es Ihnen"**, excluding the sustained vowels (a, i, u), from the SVD dataset.  
  
This analysis utilized Sklearn and Pytorch, and the environment used was Google Colab (GPU: T4).  
  
The number of files in the training and test sets is as follows:  
- train healthy   : 532 wav files    
- train_pathology : 762 wav files    
- test_healthy    : 100 wav files     
- test_pathology  : 100 wav files   


  
### Project includes 2 Analyze methods.   

## Analysis 1.   
![image](https://github.com/cksehf7/vpd-on-SVData-by-CNN-and-Sklearn/assets/132045523/9fd2dfce-f861-4ab3-8ffa-47291bc6d388)    
(image from paper :[Voice disorder detection using machine learning...](https://www.sciencedirect.com/science/article/pii/S0952197624002057))   
  
We trained the model using speech data with Scikit-Learn based on feature extraction. I referred to the paper mentioned above.  
Since we used only the phrase, the performance is not as good, achieving about 69%. In contrast, the paper reported about 99% performance (the main difference being the dataset used).  
The ~~.ipynb file contains both feature extraction and analysis.   
![image](https://github.com/cksehf7/vpd-on-SVData-by-CNN-and-Sklearn/assets/132045523/c05a1675-8055-46f6-81dd-45ae93b2b2dd)   
   
## Analysis 2.   
Voice data can be transformed into spectrograms, mel-spectrograms, and MFCCs. (Detailed parameter settings for image creation are not covered here.)  
For training, we fit 20 EfficientNet-b0 models and averaged the prediction lists to mimic a form of hard voting.   
As a result, the highest accuracy achieved was 84.5%.  
EfficientNet-b0 was chosen based on personal experience, as it was judged to be the fastest and most accurate for this task among single pretrained models like ResNet18, ResNet34, and EfficientNet-b1.  

Use the following .ipynb to create an image file: make_spec, make_mel-spec, and make_mfcc      
Use ensemble.ipynb for analysis   
![image](https://github.com/cksehf7/vpd-on-SVData-by-CNN-and-Sklearn/assets/132045523/2ce0b244-928e-47e5-90b7-4f2286da44f1)   
![image](https://github.com/cksehf7/vpd-on-SVData-by-CNN-and-Sklearn/assets/132045523/d3a81d87-5315-4255-a81c-77ef4ae8c6f9)


