#!/bin/bash

# Check if the file path is provided
if [ -z "$1" ]; then
	echo "Usage: $0 <file_path> [-i suffix]"
	exit 1
fi

file_path=$1
backup_suffix=".ori"

# parse options
shift
while getopts "i:" opt; do
	case $opt in
	i) backup_suffix=$OPTARG ;;
	*)
		echo "Invalid option: -$OPTARG" >&2
		exit 1
		;;
	esac
done

# backup the original file
backup_file="${file_path}${backup_suffix}"
cp "$file_path" "$backup_file"

# convert the JSON file to text format
jq '[.[] | "Question: \(.text)\nAnswer: \(.labels)"]' "$backup_file" >"$file_path"

echo "Conversion complete. Original file backed up as $backup_file."
