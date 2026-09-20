*This project has been created as part of the 42 curriculum by tsanjara.*

# CALL ME MAYBE

## Description

This project is part of Milestone 3 at 42 School. It introduces function calling in Large Language Models (LLMs). The project focuses on translating natural language into a language that computers can understand and execute.

### LLM

LLM stands for Large Language Model. It acts as a bridge between natural language and function calls, translating the user's request into a structured function call. It determines which function best matches the user's prompt and provides the required arguments for that function.

Function calling allows the LLMs to interact with external tools and systems by translating natural language request into structured fuction calls.

### Goal
So by means, the Goal of this project is to understand on how the LLMs works with the function calling to translate natural language requests into structured and executable action

## Instructions

Here are some instructions that may help during the test.

Make sure you have at least **11 GB of free disk space** before starting. This is needed for the project dependencies and cache files.

* `make install`: Run this the first time to install all required dependencies and prepare the caches.
* `make run`    : Run the program from the source.
* `make clean`  : Remove temporary files such as `__pycache__`.
* `make fclean` : Run this when you are completely done to remove the dependencies and caches, as they can take up a significant amount of disk space.
* `make lint`   : Check that the project follows the coding standards using **Flake8** and **Mypy**.


## Resources

* Thanks to the 42 School learning system, particularly the peer-to-peer methodology, which greatly helped in understanding the project's objectives and expectations.
* Online documentation(geeksforgeeks, w3schools, ...) and youtube tutorials, which provides ideas and inspiration for Algorithms, strategies, and implementation technics

### Use of AI

* AI tools were used as learning aids to better understand expected behaviors, clarify edge cases, and verify reasoning while debugging.
* AI was not used to directly generate the project solution, but to assist with explanations, documentation writing, and overall project organization.

## Algorithm

### Algorithm Overview

The constrained decoding approach ensures that the LLM can only generate outputs that follow the expected format.

At each step, the model generates a probability distribution for the possible next tokens. The decoder then filters out tokens that would violate the required constraints. Only valid tokens remain available for selection.

This process is repeated for each token until the complete output is generated. As a result, the final output follows the required structure and can be safely interpreted as a function call.


### Design decisions

As it was the toughest part of the project, I decided to devide it into multiple small function with each part has its own responsible
These are all of their task:
| Function | Description |
|---|---|
| `extract_only_expected` | Extracts only the JSON object that matches the expected output format. |
| `get_best_valid_token` | Selects the highest-scoring token from the valid tokens. |
| `build_json_valid_id` | Identifies all tokens that are valid for the expected output. |
| `load_vocabulary` | Loads the tokenizer vocabulary from the model. |
| `choose_function` | Converts all available functions into a readable description. |
| `build_system_prompt` | Tells the LLM which functions are available and enforces the expected JSON format. |

The system prompt also take an important role on choosing the functions and prevents from choosing an unmatched function and set the name of function as ""None

### Performance analysis

The implementation uses a greedy decoding strategy, by selecting the highest ligit with valid token at each step. This keeps the decoding process simple and deterministic.
The vocabulary is filtered once into a set of valid token IDs, allowing token selection to operate only permitted tokens.

### Challenges faced

Facing a new project is always tough, especially when trying to understand the goal of the project, the concepts behind it, and what the subject expects.

After understanding what needs to be done, it is still difficult to implement it and make the code work according to the requirements of the subject.

One of the main challenges throughout this project was understanding how an LLM's token prediction process can be controlled using constraints. As the subject itself says, the function calls must match the user's prompt, and the function name must be set to `None` when there is no matching function for the user's request.

Another challenge was finding a practical approach to control the model's output and make it generate the expected JSON structure while still respecting the constraints.

I solved these challenges by first breaking the project into smaller parts and trying to understand each concept separately. I also discussed the difficulties with my teammate and used more tests to understand how the model behaved with different prompts and function calls.

Through these discussions and tests, I was finally able to find an approach that works for the project. It may not be a perfect or complete solution to the general problem of controlling an LLM, but it was a practical approach that allowed me to meet the requirements of this project and better understand how constrained decoding and function calling work.

### Testing Strategy

I validated my implementation by testing it with different prompts and function definitions to check how the model behaves in different situations.

I tested prompts that clearly match an available function, as well as prompts that do not match any available function. In the second case, I verified that the result correctly returns `None` instead of selecting an unrelated function.

I also tested different types of arguments to make sure they are correctly extracted and converted according to the expected parameter types.

For the constrained decoding part, I tested different prompts to verify that the model generates a valid JSON structure and that the generated result can be correctly parsed and processed by the program.

Finally, I tested edge cases such as invalid arguments, missing parameters, unsupported requests, and prompts containing special characters. These tests helped me identify problems in the implementation and improve the reliability of the final result.


### Example usage

As mentioned in the instructions above, make sure that all the required dependencies are installed before running the program.

The project also requires a directory containing the necessary data files. These files are used by the program to find the function definitions and the files containing the user prompts.

The project also uses `llm_sdk`, which provides the LLM used by the program.

Once everything is properly set up, the program can be launched with:

```bash
make run
```

The program will load the function definitions and user prompts, process the prompts using the LLM, and generate the corresponding function calls.

Make sure that each function definition contains all the required parameters:

* `name`
* `description`
* `parameters`
* `returns`

Do not add extra parameters. Only these four parameters are allowed in the function definition.

For example:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {
        "type": "number"
      },
      "b": {
        "type": "number"
      }
    },
    "returns": {
      "type": "number"
    }
  },
  {
    "name": "fn_reverse_string",
    "description": "Reverse a string and return the reversed result.",
    "parameters": {
      "s": {
        "type": "string"
      }
    },
    "returns": {
      "type": "string"
    }
  }
]
```

For example, the prompt `Add 5 and 10` can be matched with `fn_add_numbers`, while `Reverse "hello"` can be matched with `fn_reverse_string`.

