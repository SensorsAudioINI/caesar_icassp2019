function autorecord_priors()
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
train_set = readtable('../train_set.txt', 'Delimiter', ' ', 'ReadVariableNames', false);
test_set = readtable('../test_set.txt', 'Delimiter', ' ', 'ReadVariableNames', false);

u = udp('localhost',8997,'timeout',60);

%for j = 1:height(train_set)
ang = 0:10:180;
trackID = train_set{1,5}{1};

for j = ang        
    %Open connection to jAER:
    fopen(u);
    a =  ['startlogging ' destDat 'prior_long_hv_' num2str(j) '.aedat'];
    cmd = sprintf('%s', a);
    
    fprintf(u,cmd);
    fprintf('%s',fscanf(u));
    pause(0.2);
    
    % file 1
    filename1 = train_set{48, 1}{1};
    [ts1, sr] = readnist([baseDir filename1]);    
    
    %check for message to comeback, %try catch
    
    x=(ts1-min(ts1))*2/(max(ts1)-min(ts1))-1; %2 is the range from 1 to -1
    % delays
    [x, y, ~] = delays4priors(x, sr, j);
    
    Y = [x, y];
    Y = [Y; Y; Y; Y; Y; Y; Y; Y; Y; Y];
    Y=(Y - min(min(Y)))*2/(max(max(Y))-min(min(Y)))-1;
    
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

