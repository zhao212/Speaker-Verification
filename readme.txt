EE 214A
====================
Project Readme file
Authors: Wei Qian, Jiaoyang Li, Qiyue Zhao, Gena Xie

-------------------------------------------------------------------------
Running Instructions
1. Add folders "voicebox" to the Matlab path
2. Run silenceRemover.m£¬ change the data input path if needed. This function will put the edited files in WavDataEdit folder
3. Run VoiceSauce:
	- In parameter estimation, set input directory to the FEMALE_SET214A17 in original data	
	- Run feature "H1*-H2*, H2*-H4*" "CPP" and save to Features/feature_data/FEMALE_SET214A17
	- Set input directory to the FEMALE_SET214A17 in WavDataEdit	
	- Run feature "F0 (straight)" and save to Features/F0/FEMALE_SET214A17
	- Run to generate features and do the same steps to MALE_SET214A17	
4. Run sampleCode.m
--------------------------------------------------------------------------