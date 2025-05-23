import os
from typing import Dict, Any
from PIL import Image
import ast
import io
import base64
from io import BytesIO
import numpy as np
import json
import requests
import torch
import openai
from tqdm import tqdm
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
os.environ['HF_HOME'] = '<cache-dir>' # if available

class GPT4VInference:

  def __init__(
      self,
      dataset_path: str,
      api_key: str = None,
      model_name: str = "gpt-4o",
      max_tokens: int = 512,
      rate_limit: int = 10,       # requests per second
      http_timeout: int = 60,     # seconds
      max_retries: int = 2
  ):
      """
      Initialize GPT-4o for multimodal inference with concurrent API calls

      Args:
          dataset_path (str): Path to dataset JSON
          api_key (str): OpenAI API key
          model_name (str): OpenAI model name
          max_tokens (int): Maximum tokens to generate
          rate_limit (int): Maximum API calls per second
          http_timeout (int): HTTP timeout in seconds
          max_retries (int): Maximum number of retries for API calls
      """
      self.dataset_path = dataset_path
      self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
      if not self.api_key:
          raise ValueError("OpenAI API key missing")
      
      openai.api_key = self.api_key
      self.model_name = model_name
      self.max_tokens = max_tokens
      self.rate_limit = rate_limit
      self.http_timeout = http_timeout
      self.max_retries = max_retries
      
      # Initialize prompts
      self.scene_graph_prompt_template = self._scene_graph_prompt()
      self.intent_prompt_template = self._intent_prompt()
      self.all_gen_prompt_template = self._all_gen_prompt()

  def load_data(self):
    with open(self.dataset_path, 'r') as f:
      return json.load(f)

  def _scene_graph_prompt(self):
    scene_graph_prompt = '''
    For the provided image generate a scene graph in JSON format that includes the following, be consise and consider only important objects:
    1. Objects in the frame. The special requirement is that you must incude every pedestrian and cyclists separately and not group them as people or cyclists.
    2. Object attributes inside object dictionary that are relevant to answering the question. Object attribute should include the state of the object eg. moving or static, description of the object such as color, orientation, etc.
    3. Object bounding boxes. These should be with respect to the original image dimensions.
    3. Object relationships between objects. This should be detailed of upto 4 words.

    Limit your response to only at most 5 most relevant objects in the scene.

    an example structure would look like this: {"Objects": {"name_of_object": { "attributes": [], "bounding_box": []}}, "Relationships": {"from": "name_of_object1", "to": "name_of_object2", "relationship": relationship between obj_1 and obj_2}}
    Strictly output in valid JSON format.
    Scene Graph:
    '''
    return scene_graph_prompt

  def _intent_prompt(self):
    intent_prompt = """
    For the provided scene graph, image and question, generate a object-intent JSON which includes the following:
    1. All objects from the scene graph.
    2. Predicted intent for every object. Intent should be one of these values:
    2.1 Lateral (Sideways) Intent Options (has to be from these two options):",
                "   - 'goes to the left'",
                "   - 'goes to the right'\n",

                "2.2. Vertical Intent Options:",
                "   - 'moves away from ego vehicle'",
                "   - 'moves towards ego vehicle'",
                "   - 'stationary'\n",

    3. Reason for this prediction.
    4. Bounding box of the object. these should be with respect to orginal image dimensions.

    an example structure would look like this: {"name_of_object1": {"Intent": "predicted intent", "Reason": "reason for this prediction", "Bounding_box": [x1, y1, x2, y2] }}
    Strictly output in valid JSON format.
    """
    return intent_prompt

  def _all_gen_prompt(self):
    all_gen_prompt = """
    For the provided image and question, generate a object-intent JSON which includes the following:
    1. At most 5 objects from the scene including Pedestrians and Cylists.
    2. Predicted intent for every object. Intent should be one of these values:
    2.1 Lateral (Sideways) Intent Options (has to be from these two options):",
                "   - 'goes to the left'",
                "   - 'goes to the right'\n",

                "2.2. Vertical Intent Options:",
                "   - 'moves away from ego vehicle'",
                "   - 'moves towards ego vehicle'",
                "   - 'stationary'\n",

    3. Risk score for this prediction (Yes or No)
    4. Bounding box of each object. these should be with respect to orginal image dimensions.
    5. Suggested action given the scene and risk score
    
    an example structure would look like this for given scene dictionary of {"Risk": Yes/No, "Suggested_action": "suggested action for ego vehicle", "name_of_object": {"Intent": ["predicted lateral intent", "predicted vertical intent"], "Reason": "reason for this prediction", "Bounding_box": [x1, y1, x2, y2] }} for all objects and NOT a list
    The Intent field list should ALWAYS have two values: one for lateral and one for vertical. Strictly output in valid JSON format. This JSON shoud NOT contain details from the scene graph such as the relationships or attributes, stick to the format mentioned above.
    """
    self.all_gen_prompt_template = self._all_gen_prompt()
    return all_gen_prompt

  def encode_image_to_base64(self, image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    width, height = img.size
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8'), height, width
    
  def extract_and_fix_json(self, raw: str, prompt_type: str) -> dict:
        """
        1) Extract exactly one balanced JSON object from raw text.
        2) Try to parse raw -> dict.
        3) On failure apply regex fixes (quotes, commas, braces).
        4) On second failure, send to GPT-4o-mini to correct JSON.
        """
        import re
        import json

        def _extract_balanced(s: str) -> str:
            # Find the first balanced {...} block
            start = s.find('{')
            if start < 0:
                return s
            depth = 0
            for idx, ch in enumerate(s[start:], start):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return s[start: idx + 1]
            # Fallback: return from first brace to end
            return s[start:]

        def _basic_fix(text: str) -> str:
            # Normalize smart quotes
            text = text.replace('“', '"').replace('”', '"')
            text = text.replace('‘', "'").replace('’', "'")
            # Strip markdown fences
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]
            # Python literals -> JSON
            text = re.sub(r"\bTrue\b", 'true', text)
            text = re.sub(r"\bFalse\b", 'false', text)
            text = re.sub(r"\bNone\b", 'null', text)
            # Single -> double quotes for keys and values
            text = re.sub(r"'(\w+)'\s*:", r'"\1":', text)
            text = re.sub(r":\s*'([^']*)'", r': "\1"', text)
            # Remove trailing commas
            text = re.sub(r",\s*(?=[}\]])", '', text)
            # Balance braces
            opens, closes = text.count('{'), text.count('}')
            if opens > closes:
                text += '}' * (opens - closes)
            elif closes > opens:
                text = '{' * (closes - opens) + text
            return text

        # 1) Optional sentinel extraction
        m = re.search(r"<BEGIN_JSON>(.*?)<END_JSON>", raw, re.S)
        fragment = m.group(1).strip() if m else raw.strip()

        # 2) Pull out one balanced JSON block
        candidate = _extract_balanced(fragment)

        # 3) First parse attempt
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            # print(f"First parse error: {e}")
            pass

        # 4) Apply regex-based fixes
        fixed = _basic_fix(candidate)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            # print(f"After basic fix parse error: {e}")
            pass

        # 5) Fallback to GPT-4o-mini
        schema = self._scene_graph_prompt() if prompt_type == 'scene' else self._intent_prompt()
        # print('Fallback to gpt-4o-mini for fixing JSON...')
        system = 'You are a JSON formatter. Output only valid JSON.'
        user = (
            f"The following JSON is invalid. Your job is to only change the structural elements of this task and not change the content. Fix it to match this schema exactly, preserving structure:\n\n"
            f"{schema}\n\nBroken JSON:\n```json\n{fixed}\n```"
        )
        resp = openai.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            temperature=0
        )
        corrected = resp.choices[0].message.content
        # Strip fences and parse
        if corrected.startswith('```json'):
            corrected = corrected[7:]
        if corrected.endswith('```'):
            corrected = corrected[:-3]
        return json.loads(corrected)
  
  def process_image_for_scene_graph(self, image_url):
    try:
      # Get image dimensions
      b64, height, width = self.encode_image_to_base64(image_url)
      
      # Create prompt with dimensions
      scene_graph_prompt = f"{self.scene_graph_prompt_template}\nOriginal dimensions: height={height}, width={width}"
      
      # Prepare API call
      messages = [{
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
          {"type": "text", "text": scene_graph_prompt}
        ]
      }]
      
      # Make API call with retries
      for attempt in range(self.max_retries):
        try:
          result = openai.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0,
            timeout=self.http_timeout
          )
          
          # Extract and process response
          output_text = result.choices[0].message.content
          
          # Clean up markdown formatting
          if output_text.startswith("```json"): output_text = output_text[7:]
          if output_text.endswith("```"): output_text = output_text[:-3]
          
          # Extract and fix JSON
          scene_graph = self.extract_and_fix_json(output_text, prompt_type="scene")
          return scene_graph
          
        except Exception as e:
          if attempt < self.max_retries - 1:
            print(f"Retrying scene graph generation after error: {e}")
            time.sleep(1)  # Wait before retry
          else:
            print(f"Failed to generate scene graph after {self.max_retries} attempts: {e}")
            return {}
    except Exception as e:
      print(f"Error processing image for scene graph: {e}")
      return {}

  def process_image_for_intent(self, image_url, scene_graph):
    try:
      # Get image dimensions
      b64, height, width = self.encode_image_to_base64(image_url)
      
      # Create prompt with dimensions and scene graph
      intent_prompt = f"{self.intent_prompt_template}\nOriginal dimensions: height={height}, width={width}.\nScene graph: {json.dumps(scene_graph)}"
      
      # Prepare API call
      messages = [{
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
          {"type": "text", "text": intent_prompt}
        ]
      }]
      
      # Make API call with retries
      for attempt in range(self.max_retries):
        try:
          result = openai.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0,
            timeout=self.http_timeout
          )
          
          # Extract and process response
          output_text = result.choices[0].message.content
          
          # Clean up markdown formatting
          if output_text.startswith("```json"): output_text = output_text[7:]
          if output_text.endswith("```"): output_text = output_text[:-3]
          
          # Extract and fix JSON
          intent_json = self.extract_and_fix_json(output_text, prompt_type="intent")
          return intent_json
          
        except Exception as e:
          if attempt < self.max_retries - 1:
            print(f"Retrying intent generation after error: {e}")
            time.sleep(1)  # Wait before retry
          else:
            print(f"Failed to generate intent after {self.max_retries} attempts: {e}")
            return {}
    except Exception as e:
      print(f"Error processing image for intent: {e}")
      return {}

  def process_image_one_pass(self, image_url):
    try:
      # Get image dimensions
      b64, height, width = self.encode_image_to_base64(image_url)
      
      # Create prompt with dimensions and scene graph
      intent_prompt = f"{self.all_gen_prompt_template}\nOriginal dimensions: height={height}, width={width}.\n Analyse this image:"
      
      # Prepare API call
      messages = [{
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
          {"type": "text", "text": intent_prompt}
        ]
      }]
      
      # Make API call with retries
      for attempt in range(self.max_retries):
        try:
          result = openai.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0,
            timeout=self.http_timeout
          )
          
          # Extract and process response
          output_text = result.choices[0].message.content
          
          # Clean up markdown formatting
          if output_text.startswith("```json"): output_text = output_text[7:]
          if output_text.endswith("```"): output_text = output_text[:-3]
          
          # Extract and fix JSON
          intent_json = self.extract_and_fix_json(output_text, prompt_type="intent")
          return intent_json
          
        except Exception as e:
          if attempt < self.max_retries - 1:
            print(f"Retrying intent generation after error: {e}")
            time.sleep(1)  # Wait before retry
          else:
            print(f"Failed to generate intent after {self.max_retries} attempts: {e}")
            return {}
    except Exception as e:
      print(f"Error processing image for intent: {e}")
      return {}

  def _process_frame(self, frame_id, frame_data, raw = False):
    """Process a single frame to generate scene graph and intent"""
    try:
      url = frame_data["image_path"]
      if raw:
          combined_res = self.process_image_one_pass(url)
          return frame_id, combined_res, None
      sg = self.process_image_for_scene_graph(url)
      intent = self.process_image_for_intent(url, sg)
      return frame_id, sg, intent
    except Exception as e:
      print(f"Error processing frame {frame_id}: {e}")
      return frame_id, {}, {}

  def run_inference(self, limit=None, overwrite=True, raw = False):
    """Run inference on dataset with concurrent API calls"""
    data = self.load_data()
    out_dir = os.path.join(os.path.dirname(self.dataset_path), "outputs", "gpt4v")
    os.makedirs(out_dir, exist_ok=True)
    # Prepare items to process
    items = list(data.items())
    print(f'Original num items {len(items)}')
    if limit:
        items = items[:limit]
    if raw:
        raw_path = os.path.join(out_dir, "all_raw_op.json")
        if not overwrite and os.path.exists(raw_path):
          with open(raw_path) as f: all_raw = json.load(f)
        else:
          all_raw = {}
        if not overwrite:
            items = [
                (fid, fdata)
                for fid, fdata in items
                if fid not in all_raw
            ]
    else:
        sg_path = os.path.join(out_dir, "all_scene_graphs.json")
        intent_path = os.path.join(out_dir, "all_intent_jsons.json")
    
        # Load existing data if not overwriting
        if not overwrite and os.path.exists(sg_path):
          with open(sg_path) as f: all_sg = json.load(f)
        else:
          all_sg = {}
        if not overwrite and os.path.exists(intent_path):
          with open(intent_path) as f: all_intent = json.load(f)
        else:
          all_intent = {}
    
    
        if not overwrite:
            items = [
                (fid, fdata)
                for fid, fdata in items
                if fid not in all_sg or fid not in all_intent
            ]
    print(f'Remaining num items: {len(items)}')
    # max 2 requests per frame → concurrency = rate_limit // 2
    concur = max(1, self.rate_limit // 2)
    chunks = [items[i:i+concur] for i in range(0, len(items), concur)]
    
    # Process frames concurrently
    with ThreadPoolExecutor(max_workers=concur) as exe:
      processed = 0
      for batch in tqdm(chunks, desc="Processing batches"):
        futures = {exe.submit(self._process_frame, fid, fd, raw): fid for fid, fd in batch}
        
        for fut in as_completed(futures):
          if raw:
              fid, rw, _ = fut.result()
              if rw:  # Only save if both were generated successfully
                # print(f"\nFrame {fid}:")
                print(f"Raw_output: {json.dumps(rw, indent=4)}")
                all_raw[fid] = rw
                processed += 1   
          else:
              fid, sg, intent = fut.result()
              if sg and intent:  # Only save if both were generated successfully
                # print(f"\nFrame {fid}:")
                # print(f"Scene graph: {json.dumps(sg, indent=4)}")
                # print(f"Intent: {json.dumps(intent, indent=4)}")
                all_sg[fid] = sg
                all_intent[fid] = intent
                processed += 1
        if raw:
            with open(raw_path, "w") as f: json.dump(all_raw, f)
        else:
            # Save after each batch
            with open(sg_path, "w") as f: json.dump(all_sg, f)
            with open(intent_path, "w") as f: json.dump(all_intent, f)
        
        # Respect rate limiting
        time.sleep(1)
    
    print(f"Done! Processed {processed} frames.")
    if raw:
        return all_raw, None
    return all_sg, all_intent

if __name__ == "__main__":
  # Set your API key here or in environment variables
  api_key='<openai-api-key>'
  # openai.api_key = api_key
  
  # Initialize the inference class with configuration
  gpt4v_inference = GPT4VInference(
    dataset_path='<dataset-path>',
    model_name="gpt-4o-mini",
    max_tokens=512,
    rate_limit=6,  # Adjust based on your API tier
    http_timeout=60,
    max_retries=2,
    api_key = api_key
  )
  
  # Run inference with limit for testing
  scene_graph, intent_json = gpt4v_inference.run_inference(overwrite = True, limit=4, raw = True)
  
  print("Completed inference!")