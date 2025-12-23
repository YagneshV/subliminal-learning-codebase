import json
import random
import sys

def shuffle_jsonl_completions(input_filepath, output_filepath):
    """
    Reads a JSONL file, shuffles the numbers in the 'completion' field
    of each line, and writes the modified data to a new JSONL file.
    
    The script automatically detects if the numbers are separated by
    commas, semicolons, newlines, spaces, or if they are in a JSON list
    or a parenthesized tuple-like string.

    Args:
        input_filepath (str): The path to the input JSONL file.
        output_filepath (str): The path to the output JSONL file.
    """
    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile, \
             open(output_filepath, 'w', encoding='utf-8') as outfile:
            
            print(f"Reading from '{input_filepath}'...")
            
            for line_number, line in enumerate(infile, 1):
                try:
                    # Parse the JSON object from the current line
                    data = json.loads(line)
                    completion_string = data.get("completion", "").strip()

                    if completion_string:
                        numbers_list = []
                        delimiter = ''

                        # First, check for a JSON list format
                        if completion_string.startswith('[') and completion_string.endswith(']'):
                            # Remove the brackets and split by comma
                            numbers_string = completion_string[1:-1]
                            numbers_list = [n.strip() for n in numbers_string.split(',') if n.strip()]
                            delimiter = ', '
                        
                        # Check for a parenthesized format like (1, 2, 3)
                        elif completion_string.startswith('(') and completion_string.endswith(')'):
                            # Remove the parentheses and split by comma
                            numbers_string = completion_string[1:-1]
                            numbers_list = [n.strip() for n in numbers_string.split(',') if n.strip()]
                            delimiter = ', '
                        
                        # If not a list or parenthesized string, check for other delimiters
                        else:
                            # Prioritize more specific delimiters first
                            if ', ' in completion_string:
                                delimiter = ', '
                            elif '; ' in completion_string:
                                delimiter = '; '
                            elif ';' in completion_string:
                                delimiter = ';'
                            elif ',' in completion_string:
                                delimiter = ','
                            elif '\'' in completion_string:
                                delimiter = '\''
                            elif '\n' in completion_string:
                                delimiter = '\n'
                            else:
                                # Fallback to space as the delimiter
                                delimiter = ' '

                            numbers_list = [n.strip() for n in completion_string.split(delimiter) if n.strip()]

                        if not numbers_list:
                            print(f"Warning: No valid numbers found in completion on line {line_number}. Skipping.")
                            outfile.write(json.dumps(data) + '\n')
                            continue

                        # Shuffle the list of number strings in-place
                        random.shuffle(numbers_list)

                        # Re-join the shuffled numbers with the determined delimiter
                        if completion_string.startswith('(') and completion_string.endswith(')'):
                            # Special handling for the parenthesized format
                            shuffled_completion = f"({delimiter.join(numbers_list)})"
                        elif completion_string.startswith('[') and completion_string.endswith(']'):
                            # Special handling for JSON list format
                            shuffled_completion = f"[{delimiter.join(numbers_list)}]"
                        else:
                            shuffled_completion = delimiter.join(numbers_list)
                        
                        # Update the 'completion' field in the data dictionary
                        data["completion"] = shuffled_completion

                    # Write the modified JSON object to the output file
                    outfile.write(json.dumps(data) + '\n')

                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON on line {line_number}. Skipping.")
                    # Write the original line to the output file for manual inspection
                    outfile.write(line)
                
            print(f"Process complete. Shuffled data saved to '{output_filepath}'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python shuffle_jsonl.py <input_file.jsonl> <output_file.jsonl>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        shuffle_jsonl_completions(input_file, output_file)
