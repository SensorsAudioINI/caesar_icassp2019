function [allAddr, allTs] = WRecDigitTESTWOMANCochleaAMS1c(doRecord,volumes, DeltaFlag, recordFileName)
% WRecDigitTESTWOMANCochleaAMS1c Summary of this function goes here
% For recording files in a database
% Author S-C. Liu (2016) Modified 20.06.2018
% Call the function without arguments to plot a previously stored freqResponse
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Otherwise:
% calibrationname is a string used for the foldername to store the files
% frequencies is a vector containing the frequencies to test in Hz. Example: 10.^(2:0.2:4)
% volumes is a vector containing the volume levels to play. Example: 0.1:0.1:0.4)
% [allAddr, allTs] = WRecDigitTESTWOMANCochleaAMS1c(1,[0.1], 1, 'TestTIDIGIT2')
% signallength is the playtime of every frequency (in seconds)
% doRecord=0 -> don't record
% doRecord=1; DeltaFlag=0; volumes=[0.3], recordFileName='Recording1.txt'

%filesepo= '/'; %replace filesep
baseDir='C:\Users\shih\Documents\cochleadata\TIDIGITS\';
destDir=['C:\Users\shih\Documents\cochleaamsworkfiles\cochleaAMSc\RecordingTIDIGITTestWoman' recordFileName '\'];

mkdir(destDir);

fileID = fopen([destDir 'log.txt'],'a');

% ENEA
train_set = readtable('train_set.txt', 'Delimiter', ' ');
test_set = readtable('test_set.txt', 'Delimiter', ' ');

u = udp('localhost',8997,'timeout',60);

for j = 1:height(train_set)
    
    trackID = num2str(train_set{j,4});
        
    %Open connection to jAER:
    fopen(u);
    
    cmd = ['startlogging ' destDir trackID '.aedat'];
    
    fprintf(u,cmd);
    fprintf('%s',fscanf(u));
    pause(0.2);
    
    % file 1
    filename1 = train_set{j, 1}{1};
    [ts1,~] = readnist([baseDir filename1]);
    % file 2
    filename2 = train_set{j, 3}{1};
    [ts2,sr] = readnist([baseDir filename2]);
    
    %check for message to comeback, %try catch
    
    x=(ts1-min(ts1))*2/(max(ts1)-min(ts1))-1; %2 is the range from 1 to -1
    y=(ts2-min(ts2))*2/(max(ts2)-min(ts2))-1;
    
    % delays
    [x, y, ang1, ang2] = random_delays(x, y, sr);
    fprintf(fileID, '%s %f %f \n', trackID, ang1, ang2);
    
    Y = [x', y'];
    
    sound(Y,sr);
    signallength = length(Y)/sr;
    pause(signallength+1);
    fprintf(u,'stoplogging');
    fprintf('%s',fscanf(u));
        
%     [allAddr, allTs]=loadaerdat([dirRec speakerID '\' outName '.aedat']);

    fclose(u);
    pause(0.5);
end

% clean up the UDP connection to jAER:

delete(u);
clear u

fclose(fileID);

