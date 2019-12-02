function [y] = Latex_table(A, row_index, col_index, top_left, label, caption, file_name)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Output a matrix as a Latex table
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [m,n] = size(A);
    fileID = fopen(file_name,'a');

    fprintf(fileID, '\n\n');
    fprintf(fileID, '\n\n');
    fprintf(fileID, '\\begin{table}[H] \n' );
    fprintf(fileID, '\\centering\n');
    fprintf(fileID, '\\label{');
    fprintf(fileID, label);
    fprintf(fileID, '}\n');
    fprintf(fileID, '\\begin{tabular}{c');
    for k = 1:n
        fprintf(fileID, 'c');
    end
    fprintf(fileID, '}\n');
    fprintf(fileID, '\\toprule\n');
    fprintf(fileID, top_left);
    fprintf(fileID, ' &');
    for k = 1:n
        fprintf(fileID, col_index{k});
        if k<n
            fprintf(fileID, ' &');
        end
    end
    fprintf(fileID,'\\\\ \n');
    fprintf(fileID,'\\midrule \n');

    for i =1:m
        fprintf(fileID, row_index{i});
        fprintf(fileID, '& ');
        for j=1:n
            fprintf(fileID, '%.3f ' ,A(i,j) );
            if j<n
            fprintf(fileID, ' &');
            end
        end
        fprintf(fileID, '\\\\ \n ' );
    end
    fprintf(fileID, '\\bottomrule\n' );
    fprintf(fileID, '\\end{tabular}\n');
    fprintf(fileID, '\\caption{')
    fprintf(fileID, caption);
    fprintf(fileID, '}\n');
    fprintf(fileID, '\\end{table}');
    fclose(fileID);
end

