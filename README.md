*This project has been created as part of the 42 curriculum by tsanjara.*

# CALL ME MAYBE

This project has been created as part of the 42 curriculum by tsanjara.

## Description

This project is part of Milestone 3 at 42 School. It introduces function calling in Large Language Models (LLMs). The project focuses on translating natural language into a language that computers can understand and execute.

### LLM

LLM stands for Large Language Model. It acts as a bridge between natural language and function calls, translating the user's request into a structured function call. It determines which function best matches the user's prompt and provides the required arguments for that function.

Function calling allows LLMs to interact with external tools and systems by translating natural language requests into structured function calls.

### Goal

The goal of this project is to understand how LLMs use function calling to translate natural language requests into structured and executable actions.

## Instructions

Make sure you have at least **11 GB of free disk space** before starting. This is needed for the project dependencies and cache files.

- `make install`: Run this the first time to install all required dependencies and prepare the caches.
- `make run`: Run the program from the source.
- `make clean`: Remove temporary files such as `__pycache__`.
- `make fclean`: Remove the dependencies and caches when you are completely done, as they can take up significant disk space.
- `make lint`: Check that the project follows the coding standards using **Flake8** and **Mypy**.

## Resources

- Thanks to the 42 School learning system, particularly the peer-to-peer methodology, which greatly helped in understanding the project's objectives and expectations.
- Online documentation (GeeksforGeeks, W3Schools, etc.) and YouTube tutorials provided ideas and inspiration for algorithms, strategies, and implementation techniques.

### Use of AI

- AI tools were used as learning aids to better understand expected behaviors, clarify edge cases, and verify reasoning while debugging.
- AI was not used to directly generate the project solution, but to assist with explanations, documentation writing, and overall project organization.

## Algorithm

### Algorithm Overview

The project uses a constrained decoding strategy to generate structured function calls. Instead of allowing the model to select any token from its vocabulary, the decoding process restricts token selection according to the expected output format.

At each generation step, the model assigns scores to possible next tokens. The decoding mechanism uses these scores to select a valid token, progressively constructing the expected JSON output.

This approach helps ensure that the generated function call follows the required structure and can be parsed by the program.

### Design Decisions

I divided the implementation into several functions, each responsible for a specific part of the process.

| Function | Responsibility |
|---|---|
| `extract_only_expected` | Extracts the expected JSON object from the generated text. |
| `get_best_valid_token_func` | Selects the best valid token to progressively identify the appropriate function. |
| `build_json_valid_id` | Determines which token IDs are allowed during function selection. |
| `load_vocabulary` | Loads the model's vocabulary for token processing. |
| `choose_function` | Prepares the available function definitions so the model can compare them with the user's request. |
| `build_system_prompt` | Guides the model in choosing the most suitable function and generating the expected JSON structure. |

The main objective of this process is to select the function that best matches the user's request. The model uses the available function descriptions and the system prompt to identify the appropriate function.

Once a function has been selected, the program extracts its required parameters and validates them against the corresponding parameter model. Separating function selection from parameter generation makes the implementation easier to test, debug, and maintain.

### Performance Analysis

The implementation uses a greedy decoding strategy, selecting the highest-scoring valid token at each step. This keeps the decoding process simple and deterministic.

The vocabulary is filtered into a set of valid token IDs, allowing token selection to operate only on permitted tokens during constrained function selection.

### Challenges Faced

Facing a new project is always challenging, especially when trying to understand its objectives, the underlying concepts, and the requirements of the subject.

After understanding what needed to be done, implementing the solution and making it work according to the requirements presented another challenge.

One of the main difficulties was understanding how an LLM's token prediction process can be controlled using constraints. The function call must match the user's request, and the generated output must follow the expected structure.

Another challenge was finding a practical approach to guide the model toward the appropriate function while generating valid JSON.

I addressed these challenges by breaking the project into smaller parts and studying each concept separately. I also discussed the difficulties with my teammate and performed additional tests to better understand the model's behavior with different prompts and function calls.

Through these discussions and experiments, I found an approach that works for the project. Although it may not be a complete solution to the general problem of controlling an LLM, it provided a practical way to meet the project's requirements and better understand constrained decoding and function calling.

### Testing Strategy

I validated my implementation by testing different prompts and function definitions to evaluate the model's behavior.

I tested prompts that clearly matched available functions to verify that the appropriate function was selected.

I also tested different argument types to check whether parameter values were correctly extracted and converted according to the expected types.

For the constrained decoding component, I tested various prompts to verify that the generated output followed the required JSON structure and could be parsed and processed by the program.

Finally, I tested edge cases such as invalid arguments, missing parameters, unsupported requests, and prompts containing special characters. These tests helped me identify problems in the implementation and improve the reliability of the final result.

### Example Usage

Before running the program, make sure all required dependencies are installed.

The project also requires a directory containing the necessary data files. These files are used by the program to access function definitions and user prompts.

The project uses `llm_sdk`, which provides the LLM used by the program.

Once everything is properly set up, launch the program with:

```bash
make run
```

The program loads the function definitions and user prompts, processes the prompts using the LLM, and generates the corresponding function calls.

Make sure each function definition contains all the required fields:

- `name`
- `description`
- `parameters`
- `returns`

Do not add extra fields. Only these four fields are allowed in each function definition.

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

Here are examples of user prompts:

```json
[
  {
    "prompt": "Add two random numbers"
  },
  {
    "prompt": "Pick and calculate the square root of a number"
  }
]
```