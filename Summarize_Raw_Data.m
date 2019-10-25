%% (1a) Convert RAW data World Bank Indicators into ISIpedia format
clc;clear;

% Open indicator list to transform

[~,Ind_list] = xlsread('C:\Users\TedVeldkamp\Documents\Dropbox\werk\Shared_folders_data\ISIPEDIA_IIASA\ISIPEDIA-portal\CDLX\Current_Conditions\Indicator_list.xlsx')

% Loop over indicators to generate files
for ii = 1:9 %size(Ind_list,1)
    input_file = strcat('C:\Users\TedVeldkamp\Documents\Dropbox\werk\Shared_folders_data\ISIPEDIA_IIASA\ISIPEDIA-portal\CDLX\Current_Conditions\RAW_DATA\',Ind_list(ii,1),'\',Ind_list(ii,1),'.csv');
    output_file = strcat('C:\Users\TedVeldkamp\Documents\Dropbox\werk\Shared_folders_data\ISIPEDIA_IIASA\ISIPEDIA-portal\CDLX\Current_Conditions\INTERMEDIATE_DATA\',Ind_list(ii,3),'.mat');
for u = 1
%% Setup the Import Options
opts = delimitedTextImportOptions("NumVariables", 64);

% Specify range and delimiter
opts.DataLines = [5, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["DataSource", "WorldDevelopmentIndicators", "VarName3", "VarName4", "VarName5", "VarName6", "VarName7", "VarName8", "VarName9", "VarName10", "VarName11", "VarName12", "VarName13", "VarName14", "VarName15", "VarName16", "VarName17", "VarName18", "VarName19", "VarName20", "VarName21", "VarName22", "VarName23", "VarName24", "VarName25", "VarName26", "VarName27", "VarName28", "VarName29", "VarName30", "VarName31", "VarName32", "VarName33", "VarName34", "VarName35", "VarName36", "VarName37", "VarName38", "VarName39", "VarName40", "VarName41", "VarName42", "VarName43", "VarName44", "VarName45", "VarName46", "VarName47", "VarName48", "VarName49", "VarName50", "VarName51", "VarName52", "VarName53", "VarName54", "VarName55", "VarName56", "VarName57", "VarName58", "VarName59", "VarName60", "VarName61", "VarName62", "VarName63", "VarName64"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];
opts = setvaropts(opts, [1, 2, 3, 4, 64], "TrimNonNumeric", true);
opts = setvaropts(opts, [1, 2, 3, 4, 64], "ThousandsSeparator", ",");
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
DATA = readtable(char(input_file), opts);

%% Convert to output type
DATA = table2array(DATA);
DATA2 = DATA(2:end,5:end);
DATA2_header = DATA(1,5:end);

%% Clear temporary variables
clear opts
end 

% Load country names data
[~,C_ISI] = xlsread('C:\Users\TedVeldkamp\Documents\Dropbox\werk\Shared_folders_data\ISIPEDIA_IIASA\ISIPEDIA-portal\CDLX\Current_Conditions\Countries_ISIPEDIA.xls')
[~,C_WB] = xlsread('C:\Users\TedVeldkamp\Documents\Dropbox\werk\Shared_folders_data\ISIPEDIA_IIASA\ISIPEDIA-portal\CDLX\Current_Conditions\Countries_WorldBank.xls')

% Match WB indicator data to ISIpedia country list
C_WB = C_WB(2:end,1);
C_ISI = C_ISI(2:end,1);
for i = 1:size(C_ISI,1)
    Country_name = char(C_ISI(i,1));
    
    for j = 1:size(C_WB,1)
    tf(j,1) = strcmp( C_ISI(i,1),C_WB(j,1));
    end
    [r,c] = find(tf ==1);
    
    if nansum(tf,1) > 0
    NEW_DATA_TS(i,1:size(DATA2,2)) = DATA2(r,:);
    
    else 
    NEW_DATA_TS(i,1:size(DATA2,2)) = NaN(1,60);
    end   
end

% Estimate means for period 2008-2017
NEW_DATA_MEAN = nanmean(NEW_DATA_TS(:,49:58),2);

% Save matlab output file as temporary data
save(char(output_file), 'NEW_DATA_TS', 'C_ISI', 'DATA2_header', 'NEW_DATA_MEAN');
end

%% (1b) Convert RAW data UN HDI into ISIpedia format
clc;clear;

% Load data
DATA2 = xlsread('C:\Users\TedVeldkamp\Documents\Dropbox\werk\Shared_folders_data\ISIPEDIA_IIASA\ISIPEDIA-portal\CDLX\Current_Conditions\RAW_DATA\HDI_ISIPEDIA_Countries.xlsx');
DATA2_header = DATA2(1,:);
DATA2 = DATA2(2:end,:);

% Load country names data
[~,C_ISI] = xlsread('C:\Users\TedVeldkamp\Documents\Dropbox\werk\Shared_folders_data\ISIPEDIA_IIASA\ISIPEDIA-portal\CDLX\Current_Conditions\Countries_ISIPEDIA.xls')
[~,C_HDI] = xlsread('C:\Users\TedVeldkamp\Documents\Dropbox\werk\Shared_folders_data\ISIPEDIA_IIASA\ISIPEDIA-portal\CDLX\Current_Conditions\RAW_DATA\HDI_ISIPEDIA_Countries.xlsx');

% Match WB indicator data to ISIpedia country list
C_HDI = C_HDI(1:end,1);
C_ISI = C_ISI(2:end,1);
for i = 1:size(C_ISI,1)
    Country_name = char(C_ISI(i,1));
    
    for j = 1:size(C_HDI,1)
    tf(j,1) = strcmp( C_ISI(i,1),C_HDI(j,1));
    end
    [r,c] = find(tf ==1);
    
    if nansum(tf,1) > 0 && r <190
    NEW_DATA_TS(i,1:size(DATA2,2)) = DATA2(r,:);
    
    else 
    NEW_DATA_TS(i,1:size(DATA2,2)) = NaN(1,28);
    end   
end

% Estimate means for period 2008-2017
NEW_DATA_MEAN = nanmean(NEW_DATA_TS(:,19:28),2);

% Save matlab output file as temporary background data
save('C:\Users\TedVeldkamp\Documents\Dropbox\werk\Shared_folders_data\ISIPEDIA_IIASA\ISIPEDIA-portal\CDLX\Current_Conditions\INTERMEDIATE_DATA\HDI.mat', 'NEW_DATA*', 'C_ISI', 'DATA2_header');

%% Export mean data per country and indicator to separate csv files 
clc;clear;

Indicator_list = {'CO2_EM','GDP','HDI','LND_TOTL','POP_DNST','POP_GROWTH','POP_TOTL','POV_DDAY','RUR_POP_PRCT','URB_POP_PRCT'};
for i = 1:size(Indicator_list,2)
    Ind = char(Indicator_list(1,i)); 

    input_file = strcat('C:\Users\TedVeldkamp\Documents\Dropbox\werk\Shared_folders_data\ISIPEDIA_IIASA\ISIPEDIA-portal\CDLX\Current_Conditions\INTERMEDIATE_DATA\',char(Indicator_list(1,i)),'.mat');
    load(input_file); 
    
    for ii = 1:size(C_ISI,1)
        CC = char(C_ISI(ii,1));
        
        data = NEW_DATA_MEAN(ii,1);
        output_file = strcat('C:\Users\TedVeldkamp\Documents\Dropbox\werk\Shared_folders_data\ISIPEDIA_IIASA\ISIPEDIA-portal\CDLX\Current_Conditions\INTERMEDIATE_DATA\',Ind,'\', Ind,'_',CC,'.csv');
        output_file2 = strcat('C:\Users\TedVeldkamp\Documents\Dropbox\werk\Shared_folders_data\ISIPEDIA_IIASA\ISIPEDIA-portal\CDLX\Current_Conditions\INTERMEDIATE_DATA\', Ind,'.csv');
        output_file3 = strcat('C:\Users\TedVeldkamp\Documents\Dropbox\werk\Shared_folders_data\ISIPEDIA_IIASA\ISIPEDIA-portal\CDLX\Current_Conditions\INTERMEDIATE_DATA\countrries.csv');
  
        %csvwrite(output_file,data); 
        csvwrite(output_file2,NEW_DATA_MEAN); 
        csvwrite(output_file3,char(C_ISI)); 
    end
end

 
%% Export mean data per country and indicator to separate csv files - HDI
clc;clear;

Indicator_list = {'HDI'};
for i = 1:size(Indicator_list,2)
    Ind = char(Indicator_list(1,i)); 

    input_file = strcat('C:\Users\TedVeldkamp\Documents\Dropbox\werk\Shared_folders_data\ISIPEDIA_IIASA\ISIPEDIA-portal\CDLX\Current_Conditions\INTERMEDIATE_DATA\',char(Indicator_list(1,i)),'.mat');
    load(input_file); 
    
    for ii = 1:size(C_ISI,1)
        CC = char(C_ISI(ii,1));
        
        data = NEW_DATA_MEAN(ii,1);
        output_file = strcat('C:\Users\TedVeldkamp\Documents\Dropbox\werk\Shared_folders_data\ISIPEDIA_IIASA\ISIPEDIA-portal\CDLX\Current_Conditions\INTERMEDIATE_DATA\',Ind,'\', Ind,'_',CC,'.csv');
        output_file2 = strcat('C:\Users\TedVeldkamp\Documents\Dropbox\werk\Shared_folders_data\ISIPEDIA_IIASA\ISIPEDIA-portal\CDLX\Current_Conditions\INTERMEDIATE_DATA\', Ind,'.csv');
        output_file3 = strcat('C:\Users\TedVeldkamp\Documents\Dropbox\werk\Shared_folders_data\ISIPEDIA_IIASA\ISIPEDIA-portal\CDLX\Current_Conditions\INTERMEDIATE_DATA\countrries.csv');
  
        %csvwrite(output_file,data); 
        csvwrite(output_file2,NEW_DATA_MEAN); 
        %csvwrite(output_file3,char(C_ISI)); 
    end
end

    

