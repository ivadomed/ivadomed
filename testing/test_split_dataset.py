import os
import csv


participants = []

def create_tsvfile():
    # Create participants.tsv with n participants
    n = 10
    for participant_id in range(n):
        row_participants = []
        row_participants.append('sub-00' + str(participant_id))
        # 3 different disabilities: 0, 1 or 2
        row_participants.append(str(n % 3))
        # 4 different centers: 0, 1, 2 or 3
        row_participants.append(str(n % 4))
        participants.append(row_participants)

    # # Save participants.tsv
    with open("participants.tsv", 'w') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(["participant_id", "disability", "institution_id"])
        for item in sorted(participants):
            tsv_writer.writerow(item)