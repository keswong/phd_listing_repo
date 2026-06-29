import os
import json
import pandas as pd
from typing import Optional, Any
from abc import ABC, abstractmethod
from openai import OpenAI
from dotenv import load_dotenv

# Types
from pydantic import BaseModel

class Feedback(BaseModel):
    reasoning: str
    # context: str
    # pedagogy: str
    feedback: str

class Agent(ABC):
    """Abstract base class for all agents."""
    
    _data_cache: dict = {}
    _prompt_cache: dict = {}
    
    def __init__(self, model: str = "o4-mini", verbose: bool = False, debug_mode: bool = False, temperature: float = 0.0):
        self.model = model
        self.verbose = verbose
        self.debug_mode = debug_mode
        self.temperature = temperature
        # Determine paths relative to this file
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.prompts_dir = os.path.join(self.base_dir, "prompts")
        # Assuming data is in the root data directory, which is up one level from agents/
        self.project_root = os.path.dirname(self.base_dir) 
        
    def _load_prompt(self, agent_name: str) -> str:
        """Load prompt template from file. Cached after first call."""
        if agent_name in Agent._prompt_cache:
            return Agent._prompt_cache[agent_name]
        
        prompt_file = os.path.join(self.prompts_dir, f"{agent_name}.txt")
        
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        
        Agent._prompt_cache[agent_name] = prompt_template
        return prompt_template

    def _load_data(self) -> dict:
        """Load all required dataframes. Cached after first call."""
        if Agent._data_cache:
            return Agent._data_cache
        
        data_cache = {}
        
        # Paths adjusted for execution from root or general usage
        # Assuming typical project structure:
        # root/
        #   agents/
        #   data/
        #     clean_data/
        #     clean_data_2/
        
        clean_data_path = os.path.join(self.project_root, "data", "clean_data")
        clean_data_3_path = os.path.join(self.project_root, "data", "clean_data_3")
        
        try:
            data_cache['df_indicators'] = pd.read_json(
                os.path.join(clean_data_path, "indicators.json"), orient="records", lines=True
            )
            data_cache['df_sequences'] = pd.read_json(
                os.path.join(clean_data_3_path, "sequences.json"), orient="records"
            )
            data_cache['df_utterances'] = pd.read_json(
                os.path.join(clean_data_3_path, "utterances.json"), orient="records"
            )
            data_cache['df_interventions'] = pd.read_json(
                os.path.join(clean_data_3_path, "interventions.json"), orient="records"
            )
            data_cache['df_subsequences'] = pd.read_json(
                os.path.join(clean_data_3_path, "subsequences.json"), orient="records"
            )
        except ValueError as e:
            # Fallback or error handling for clearer messages
            print(f"Error loading data files: {e}")
            raise
            
        # Load task description (hardcoded for now, can be made configurable)
        data_cache['task'] = """When a vehicle turns a corner, the driver controls the two front wheels using the steering wheel. \
Since the rear wheels do not pivot, they do not follow the same path as the front wheels. \
As a result, the rear wheels will travel a smaller radius than its front wheels.

The diagram below shows a vehicle that is making a left turn. \
Both the front and rear wheels follow a circular path about the centre O. \
The front wheel travels a circular path of radius r metres. \
The difference in distance between the radius of the circular path for the front left wheel \
and the radius of the circular path of the rear left wheel is x metres. The distance between the \
front and rear wheel is called the wheelbase, w.

(a)\tShow that x^2 -2rx + w^2 = 0.
"""
        
        data_cache['task_solution'] = """In the diagram, the length of the vehicles is always at right angle to the radius of the circular path. \
This allows us to form a right-angled triangle, where the distance from centre 0 to the front left wheel is the hypotenuse.

Since the front left wheel travels a circular path of radius, the hypotenuse of the right-angle triangle is r.

One of the legs of the right-angled triangle is the wheelbase of the car, w.

The other leg of the right-angled triangle is circular path of the rear left wheel. \
Since the distance between the radius of the circular path for the front left wheel \
and the radius of the circular path of the rear left wheel is x metres, \
the radius of the circular path of the rear left wheel is (r-x). \
The other leg of the right-angled triangle is (r-x).

By applying Pythagoras' theorem, w^2 + (r-x)^2 = r^2

By using algebraic expression, w^2 + r^2 - 2rx + x^2 = r^2

By simplying and rearranging algebraic terms, x^2 - 2rx + w^2= r^2 - r^2

Hence, we show that x^2 -2rx + w^2 = 0.
"""
        
        Agent._data_cache.update(data_cache)
        return data_cache

    def _generate_llm_text(self, prompt: str) -> str:
        """Generate text using OpenAI API."""
        if self.debug_mode:
            prompt = self._interactive_prompt_edit(prompt)
            
        load_dotenv()
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        try:
            # Try the custom responses endpoint first (for o4-mini)
            if hasattr(client, 'responses') and hasattr(client.responses, 'create'):
                request_payload = {
                    "model": self.model,
                    "input": prompt,
                    "temperature": self.temperature
                }
                if self.verbose:
                    print(f"REQUEST: {json.dumps(request_payload, indent=2)}")

                try:
                    response = client.responses.create(**request_payload)
                except Exception as e:
                    if "temperature" in str(e) and "not supported" in str(e):
                        # Fallback without temperature
                        del request_payload["temperature"]
                        if self.verbose:
                            print(f"REQUEST (fallback): {json.dumps(request_payload, indent=2)}")
                        response = client.responses.create(**request_payload)
                    else:
                        raise e
                
                if self.verbose:
                    # Convert response to dict if possible, or string
                    try:
                        print(f"RESPONSE: {response.model_dump_json(indent=2)}")
                    except AttributeError:
                        print(f"RESPONSE: {response}")

                result = response.output_text
            else:
                # Fallback to standard chat completions
                request_payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature
                }
                if self.verbose:
                    print(f"REQUEST: {json.dumps(request_payload, indent=2)}")

                try:
                    response = client.chat.completions.create(**request_payload)
                except Exception as e:
                    if "temperature" in str(e) and "not supported" in str(e):
                         del request_payload["temperature"]
                         if self.verbose:
                             print(f"REQUEST (fallback): {json.dumps(request_payload, indent=2)}")
                         response = client.chat.completions.create(**request_payload)
                    else:
                        raise e
                
                if self.verbose:
                     print(f"RESPONSE: {response.model_dump_json(indent=2)}")

                result = response.choices[0].message.content
                
            if self.debug_mode:
                self._print_debug_result(result)
                
            return result
        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"

    def _generate_llm_feedback(self, prompt: str) -> Feedback | str | None:
        """Generate structured feedback using OpenAI API with Pydantic parsing."""
        if self.debug_mode:
            prompt = self._interactive_prompt_edit(prompt)
            
        load_dotenv()
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        try:
            # Try structured output parsing (requires models that support it)
            try:
                request_payload = {
                    "model": self.model,
                    "messages": [{"role": "system", "content": prompt}],
                    "response_format": Feedback, 
                    "temperature": self.temperature
                }
                # response_format is a type, so json.dumps might fail on it. 
                # We'll print a simplified version for logging.
                log_payload = request_payload.copy()
                log_payload["response_format"] = "Feedback (Pydantic Model)"
                
                if self.verbose:
                    print(f"REQUEST: {json.dumps(log_payload, indent=2)}")

                completion = client.beta.chat.completions.parse(**request_payload)
            except Exception as e:
                 if "temperature" in str(e):
                    del request_payload["temperature"]
                    if self.verbose:
                        print(f"REQUEST (fallback): {json.dumps(log_payload, indent=2)}") # Reuse log_payload, inaccurate temp but safe
                    completion = client.beta.chat.completions.parse(
                        model=self.model,
                        messages=[{"role": "system", "content": prompt}],
                        response_format=Feedback,
                    )
                 else:
                     raise e
            
            if self.verbose:
                print(f"RESPONSE: {completion.model_dump_json(indent=2)}")

            message = completion.choices[0].message
            if message.parsed:
                result = message.parsed
                if self.debug_mode:
                    self._print_debug_result(result)
                return result
            else:
                print(f"API refusal: {message.refusal}")
                return None
        except AttributeError:
            # Fallback if beta.parse is not available
            try:
                request_payload = {
                    "model": self.model,
                    "messages": [{"role": "system", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": self.temperature
                }
                if self.verbose:
                    print(f"REQUEST: {json.dumps(request_payload, indent=2)}")

                response = client.chat.completions.create(**request_payload)
            except Exception as e:
                if "temperature" in str(e):
                    del request_payload["temperature"]
                    if self.verbose:
                         print(f"REQUEST (fallback): {json.dumps(request_payload, indent=2)}")
                    response = client.chat.completions.create(**request_payload)
                else:
                    raise e
            
            if self.verbose:
                 print(f"RESPONSE: {response.model_dump_json(indent=2)}")

            # Parse manually if needed
            content = response.choices[0].message.content
            data = json.loads(content)
            # Check for direct feedback fields or nested structure
            if 'reasoning' in data and 'feedback' in data:
                result = Feedback(**data)
            elif 'feedback' in data:
                # Handle nested structure if needed
                feedback_data = data['feedback']
                if 'reasoning' in feedback_data and 'feedback' in feedback_data:
                    result = Feedback(**feedback_data)
            else:
                result = None
            
            if self.debug_mode and result:
                self._print_debug_result(result)
            elif self.debug_mode:
                    self._print_debug_result(content)

            return result

        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"

    def log_verbose(self, title: str, content: Any):
        if self.verbose:
            print("=" * 80)
            print(title)
            print("=" * 80)
            print(content)
            print("\n" + "-" * 80 + "\n")

    def _interactive_prompt_edit(self, prompt: str) -> str:
        """
        Interactive debug mode: Print prompt, allow edit in temp file, and wait for confirmation.
        """
        import tempfile
        import subprocess
        
        agent_name = self.__class__.__name__
        print("\n" + "!" * 80)
        print(f"DEBUG MODE: REVIEW PROMPT FOR {agent_name.upper()}")
        print("!" * 80)
        print(prompt)
        print("-" * 80)
        
        while True:
            choice = input("\n[E]dit prompt, [C]ontinue (generate), [A]bort? [e/C/a]: ").lower().strip()
            
            if choice == 'a':
                print("Aborting...")
                exit(0)
            
            elif choice == 'e':
                # Open in editor
                with tempfile.NamedTemporaryFile(suffix=".txt", mode='w+', delete=False, encoding='utf-8') as tf:
                    tf_path = tf.name
                    tf.write(prompt)
                
                try:
                    editor = os.environ.get('EDITOR', 'vim') # Default to vim if EDITOR not set
                    subprocess.call([editor, tf_path])
                    
                    with open(tf_path, 'r', encoding='utf-8') as tf:
                        new_prompt = tf.read()
                        
                    if new_prompt.strip() != prompt.strip():
                        print("Prompt updated!")
                        prompt = new_prompt
                        # Show updated prompt
                        print("\nUPDATED PROMPT:")
                        print("-" * 80)
                        print(prompt)
                        print("-" * 80)
                    else:
                        print("No changes made.")
                        
                except Exception as e:
                    print(f"Error opening editor: {e}")
                finally:
                    if os.path.exists(tf_path):
                        os.unlink(tf_path)
            
            else:
                # Continue (default)
                print(f"Generating for {agent_name}...")
                return prompt

    def _print_debug_result(self, result: Any):
        """Print the generated result in a clear block."""
        agent_name = self.__class__.__name__
        print("\n" + ">" * 80)
        print(f"DEBUG MODE: GENERATED OUTPUT FROM {agent_name.upper()}")
        print(">" * 80)
        
        if isinstance(result, BaseModel):
            print(result.model_dump_json(indent=2))
        elif isinstance(result, (dict, list)):
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(result)
            
        print("<" * 80 + "\n")
        
        input(f"Press Enter to continue to next step (or Ctrl+C to abort)...")


    @classmethod
    def reload_prompts(cls):
        """Clear the prompt cache to reload prompts from files."""
        cls._prompt_cache = {}

    @abstractmethod
    def run(self, *args, **kwargs) -> dict:
        pass

