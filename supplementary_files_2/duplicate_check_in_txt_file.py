# Given a text file with a list of image paths, check for duplicates.

file_path = 'C:\\Users\\Jasper\\Downloads\\Master thesis\\Code\\dataloaders\\imagenet_train_filenames.txt'

### Option 1
with open(file_path, 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    print(len(lines))
    print(len(set(lines)))

### Option 2
def check_for_duplicates(file_path):
    lines_seen = set()  # Set to store unique lines
    duplicate_lines = []  # List to store duplicate lines
    
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, start=1):
            line = line.strip()  # Remove leading/trailing whitespace
            
            # If the line is not in the set, add it
            if line not in lines_seen:
                lines_seen.add(line)
            else:
                # If the line is already in the set, it's a duplicate
                duplicate_lines.append((line_num, line))
    
    return duplicate_lines

duplicates = check_for_duplicates(file_path)

if duplicates:
    print("Duplicate lines found:")
    for line_num, line in duplicates:
        print(f"Line {line_num}: {line}")
else:
    print("No duplicate lines found.")