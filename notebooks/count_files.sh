#!/bin/bash

main_folder="/home/buehlern/neocortex-nas/shared/Skelett"

echo "Scanning $main_folder"

for bodypart in "$main_folder"/*; do
	if [ -d "$bodypart" ]; then
		bodypart_name=$(basename "$bodypart")
		echo "Bodypart: $bodypart_name"

		patient_count=$(find "$bodypart" -mindepth 1 -maxdepth 1 -type d | wc -l)
		scan_count=$(find "$bodypart" -mindepth 2 -type f | wc -l)

		echo "  Patients: $patient_count"
		echo "  Scans: $scan_count"
	fi
done
