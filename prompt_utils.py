def cut_text(txt: str, cut_ratio=0.1):
    '''
    Cut cut_ratio of the text from the end 
    '''
    stripped = txt.strip('\n ')
    end_idx = min(int(cut_ratio * len(stripped)), 1)
    return stripped[:-end_idx]

def find_nth(haystack: str, needle: str, n: int) -> int:
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def parse_code(output: str, n_examples=1):
    code_start_idx = find_nth(output, 'Output:', n_examples + 1) + len('Output:') + 1                                                     
    # The last } probably signifies the end of the main function
    # We truncate the output up to it in case the model left some notes explaining the code 
    code_end_idx = output.rfind('}') 

    return output[code_start_idx : code_end_idx + 1]

def make_prompt(truncated_txt: str):
    prompt_str = '''Please complete the following source code to make it a valid C++ program that can be compiled with C++17. Make sure to only include the code in your answer
Example:
Input: 
int main() {{
    printf("Hello, 
Output:
int main() {{
    printf("Hello, World!");
}}
Task:
Input: 
{0}
Output:
'''
    return prompt_str.format(truncated_txt)

def make_sft_example(txt: str):
    truncated_txt = cut_text(txt)
    prompt = make_prompt(truncated_txt)
    return prompt + txt + '\n'
