function WRecDigitTESTWOMANCochleaAMS1c()
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
recordFileName = 'test_spikes';
%filesepo= '/'; %replace filesep
baseDir = 'G:\TIDIGITS\';
% baseDir='C:\Users\shih\Documents\cochleadata\TIDIGITS\';
destDir=['C:\Users\sensors\' recordFileName '\'];
destWav = [destDir 'wavs\'];
destDat = [destDir 'aedat\'];

mkdir(destDir);
mkdir(destWav);
mkdir(destDat);

fileID = fopen([destDir 'log.txt'], 'wt');

% ENEA
train_set = readtable('../train_set.txt', 'Delimiter', ' ');
test_set = readtable('../test_set.txt', 'Delimiter', ' ');

u = udp('localhost',8997,'timeout',60);

%for j = 1:height(train_set)
for j =1:10   
    trackID = train_set{j,5}{1};
        
    %Open connection to jAER:
    fopen(u);
    a =  ['startlogging ' destDat trackID '.aedat'];
    cmd = sprintf('%s', a);
    
    fprintf(u,cmd);
    fprintf('%s',fscanf(u));
    pause(0.2);
    
    % file 1
    filename1 = train_set{j, 1}{1};
    [ts1, sr] = readnist([baseDir filename1]);
    mkdir([destWav remove_name(filename1)])
    
    % file 2
    filename2 = train_set{j, 2}{1};
    [ts2, sr] = readnist([baseDir filename2]);
    mkdir([destWav remove_name(filename2)])
    
    %check for message to comeback, %try catch
    
    x=(ts1-min(ts1))*2/(max(ts1)-min(ts1))-1; %2 is the range from 1 to -1
    y=(ts2-min(ts2))*2/(max(ts2)-min(ts2))-1;
    
    audiowrite([destWav filename1 '.wav'], x, sr)
    audiowrite([destWav filename2 '.wav'], y, sr)
    % delays
    [x, y, ang1, ang2] = random_delays(x, y, sr);
    fprintf(fileID, '%s %f %f \n', trackID, ang1, ang2);
    
    Y = [x, y];
    Y=(Y - min(min(Y)))*2/(max(max(Y))-min(min(Y)))-1;
    audiowrite([destWav trackID '.wav'], Y, sr)
    
    fprintf(u,'zerotimestamps');
    fprintf('%s',fscanf(u));
    
    sound(Y*0.015,sr);
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

