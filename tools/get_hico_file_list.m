
%create a list_file for HICO 
set = 'test';
RESULT_FILE = strcat('hico_file_list_',set,'.txt');
ANNO_FILE = 'Users/workhard/Desktop/hico/anno_obj.mat';

load(ANNO_FILE);

fileID = fopen(RESULT_FILE,'w');

if(strcmp(set,'test') == 1)
file_len_test = size(list_test,1);
    for i=1:file_len_test
       temp = strsplit(char(list_test(i)),'.'); 
       filename = char(temp(1));
       fprintf(fileID, '%s ', filename);
       for j = 1:80
        fprintf(fileID,'%d ',anno_test(j,i));
       end
       fprintf(fileID,'\n');   
    end
end

fprintf('done\n');
fclose(fileID);