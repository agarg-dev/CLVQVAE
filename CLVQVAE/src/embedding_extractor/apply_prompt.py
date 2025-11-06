import argparse
import sys

def apply_prompt_template(input_file, output_file, prompt_file):
    """
    Reads a prompt template from a file, then reads each line from the 
    input file, wraps it in the template, and writes to the output file.
    """
    # Read the prompt template from the provided file
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read().strip()
            # Ensure the placeholder is in the template
            if '{review_text_goes_here}' not in prompt_template:
                print(f"Error: The placeholder '{{review_text_goes_here}}' was not found in {prompt_file}", file=sys.stderr)
                sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompt_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the prompt file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Using prompt template from: {prompt_file}")
    print(f"Reading reviews from: {input_file}")
    print(f"Writing formatted prompts to: {output_file}")

    try:
        with open(input_file, 'r', encoding='utf-8') as reader, \
             open(output_file, 'w', encoding='utf-8') as writer:
            
            reviews_processed = 0
            for line in reader:
                review_text = line.strip()
                if review_text: # Ensure the line is not empty
                    formatted_prompt = prompt_template.format(review_text_goes_here=review_text)
                    writer.write(formatted_prompt + '\n')
                    reviews_processed += 1
            
            print(f"Successfully processed and wrote {reviews_processed} reviews.")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a prompt template to a text file.")
    parser.add_argument("--input-file", required=True, help="Path to the input text file (one review per line).")
    parser.add_argument("--output-file", required=True, help="Path to the output file where formatted prompts will be saved.")
    parser.add_argument("--prompt-file", required=True, help="Path to the .txt file containing the prompt template.")
    
    args = parser.parse_args()
    
    apply_prompt_template(args.input_file, args.output_file, args.prompt_file)