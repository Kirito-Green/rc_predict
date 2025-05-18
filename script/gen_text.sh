#! /bin/bash


files=(
    'demo_gnn_ref.py'
)

code_text='./code.txt'
# clean file
> $code_text

for file in ${files[*]}; do
    file_path=$file
    echo "Processing $file_path"

    if [ -f $file_path ]; then
        echo "File exists"
        echo "Generating text for $file_path"
        text=$(cat $file_path)
        echo -e "$file:\n" >> $code_text
        echo "$text" >> $code_text
    else
        echo "File does not exist"
    fi
done

echo "Text generation completed. Output saved to $code_text"
