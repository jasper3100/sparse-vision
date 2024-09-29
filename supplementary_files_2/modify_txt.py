file_path = 'C:\\Users\\Jasper\\Downloads\\Master thesis\\Code\\dataloaders\\imagenet_filenames.txt'

# I want to move all lines that contain "val" into a new file

#new_file_path = 'C:\\Users\\Jasper\\Downloads\\Master thesis\\Code\\dataloaders\\imagenet_val_filenames.txt'
#new_train_path = 'C:\\Users\\Jasper\\Downloads\\Master thesis\\Code\\dataloaders\\imagenet_train_filenames_new.txt'

def remove_lines_containing_string(file_path, string_to_remove):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Filter out lines containing the specified string
    lines = [line for line in lines if string_to_remove not in line]
    
    # Write the filtered lines back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

# Specify the path to your file and the string to remove
string_to_remove = 'val'

# Call the function
remove_lines_containing_string(file_path, string_to_remove)