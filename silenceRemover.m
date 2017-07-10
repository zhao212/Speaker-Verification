close all, clear all
%% This program will remove silences from a speech recording
    % It will do it by analyzin a frame of a soundwave, if the max
    % amplitude is under the threshold 0.04, then it will set its amplitude
    % to 0 in the new signal. If the max amplitude is over the threshold,
    % it will copy it to the new signal.
    % The script will then go on a fram by frame basis deleteing all frames
    % with a 0 amplitude
    
label = load('dataLabel.mat');

dataLabel = cat(1,label.FEMALE,label.MALE);
NumPairs = size(dataLabel,1);
fileNames = unique(dataLabel(:,1:2));
nFile = size(fileNames,1);

pathToTraining = fullfile('WavData','boy.wav');

%step 1 - break signal into 0.1 s section

n = length(fileNames);

for iFile=1:nFile
	[ip,fs]=audioread(['WavData/' fileNames{iFile} ]);
	frame_duration = 0.01;
	frame_len = frame_duration*fs;
	N = length(ip);
	num_frames = floor(N/frame_len);

	new_sig=zeros(N,1);
	count =0;
	for k=1: num_frames
	    %extract frame of speech
	    frame = ip((k-1)*frame_len +1 : frame_len*k);
	    
	    %Identify silence by finding if frame max amplitude id > 0.03
	    max_val = max(frame);
	    
	    if(max_val>0.04)
		count=count+1;
		new_sig((count-1)*frame_len +1: frame_len*count)=frame;
	    end
	end
	%get rid of extra space at end.
	for k=1: num_frames
    	%extract frame of speech
    	frame = new_sig((k-1)*frame_len +1 : frame_len*k);
    
	    %Identify silence by finding if frame max amplitude id > 0.03
	    max_val = max(frame);
	    
	    if(max_val<=0)
	  	new_sig(k*frame_len:end)=[];
        	break;
    	    end
    end
    
    %fileOut=['Features/cep/' fileNames{iFile}(1:end-3) 'mat'];
    audiowrite(['WavDataEdit/' fileNames{iFile}(1:end-3) 'wav'],new_sig,fs);
    
    
end



