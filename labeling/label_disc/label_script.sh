#!/bin/bash
input="$1"
while read line


do
	path_ilabel="$2"/derivatives/labels/"$line"/anat/"$line""$4"

	path_input="$2"/"$line"/anat/"$line""$3"
	echo "$line"
	if test -f "$path_input"; then

		mkdir -p "$6"/"$line"/anat


		if test -f "$path_ilabel"; then
    		echo "$path_ilabel exist"
    		sct_label_utils -i "$path_input" -ilabel "$path_ilabel" -create-viewer 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 -o "$6"/"$line"/anat/"$line"/"$line""$4"
	
		else
			sct_label_utils -i "$path_input" -create-viewer 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 -o "$6"/"$line"/anat/"$line""$4"
		fi

		echo "$line" >> list_done.txt
	
		echo '{
  		"Author": "'"$6"'", 
  		"Label": "label_disc_manual"
			}' > "$6"/"$line"/anat/"$line""$4".json
	fi
	
done < "$input"
