# band-gap_pxrd
This repository serves as electronic source of the article "Band-gap assesment from X-ray powder diffraction using artificial intelligence", which is published in J. Appl. Cryst. 55, 1538-1548. 

This repository has the next directories:
<ul>
 <li> TOPAS_Input_file_Example: There you find the input file (*.inp) used to simulate the diffraction patterns usinng the TOPAS software package. This input file uses xrd_step_0_02051.xy, which is an auxiliary file.</li>
 <li> DataSetCreation: Contains the jupyter-notebooks used to process the simulated diffraction patterns (convert *.xy file into a numpy array). Additionally, this directory contains the notebook used to build the compositional features that served as a second input data for the CNNs </li>
 <li>CodeToTrainCNNs: contains the code used to train the CNNs. The code is executed as follows: </li>
 $python cnn1d_OneInputVector.py cnn1d_v2.inp
 
 The file cnn1d_v2.inp is an input data with information about the input_file, output_file, the data collection (csv-file), the number of outputs in the CNN the fraction reserved to validate/test, and the control_file, which defines the architecture of the CNN.
 
 <li></li>
</ul>
