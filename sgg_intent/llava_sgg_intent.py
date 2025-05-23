import os
from typing import Dict, Any
from PIL import Image
import ast
import io
import json
import requests
import torch
from tqdm import tqdm
from io import BytesIO
import numpy as np
import openai
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from itertools import islice

os.environ['HF_HOME'] = '<cache-dir>' # if available


class LlavaInference:

    def __init__(
        self,
        dataset_path: str,
        scene_graph_prompt: str = None,
        intent_prompt: str = None,
        openai_key: str = None,
        model_path: str = "llava-hf/llava-v1.6-vicuna-13b-hf",  # Using the specified LLaVA model
        device: str = "cuda",
        cache_dir: str = os.environ['HF_HOME'],
        model = None,
        processor = None,
    ):
        """
        Initialize LLaVA model for multimodal inference

        Args:
            model_path (str): Path to LLaVA model
            device (str): Device to run inference on
        """
        if model is None:
            # Load model and processor
            self.processor = LlavaNextProcessor.from_pretrained(model_path)
            self.cache_dir = cache_dir
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype=torch.float16,  # Using float16 as recommended in docs
                low_cpu_mem_usage=True,
                cache_dir=self.cache_dir
            )
        else:
            self.processor = processor
            self.model = model
        self.openai_key     = openai_key     or os.environ["OPENAI_API_KEY"]
        openai.api_key      = self.openai_key
        self.scene_graph_prompt = self.scene_graph_prompt()
        self.intent_prompt = self.intent_prompt()
        self.dataset_path = dataset_path
        self.device = device
        # Default generation config
        self.max_new_tokens = 7500
        self.all_gen_prompt_template = self.all_gen_prompt()

    def load_data(self):
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        self.data = data
        return data

    def scene_graph_prompt(self):
        scene_graph_prompt = '''
        For the provided image generate a scene graph in JSON format that includes the following, be consise and consider only important objects:
        1. Objects in the frame. The special requirement is that you must incude every pedestrian and cyclists separately and not group them as people or cyclists (!IMPORTANT)
        2. Object attributes inside object dictionary that are relevant to answering the question. Object attribute should include the state of the object eg. moving or static, description of the object such as color, orientation, etc.
        3. Object bounding boxes in [x1, x2, y1, y2] format. These should be with respect to the original image dimensions.
        3. Object relationships between objects. This should be detailed of upto 4 words.

        Limit your response to only at most 5 most relevant objects in the scene.

        an example structure would look like this: {"Objects": {"name_of_object": { 'attributes': [], "bounding_box": []}}, "Relationships": {"from": "name_of_object1", "to": "name_of_object2", "relationship": relationship between obj_1 and obj_2}}
        Strictly output in valid JSON format. Reminder: do not group object, if there are multiple pedestrians, include them separately.
        Scene Graph:
        '''
        return scene_graph_prompt

    def intent_prompt(self):
        intent_prompt = """
        For the provided scene graph, image and question, generate a object-intent JSON which includes the following:
        For every object in scene graph, .
        1. Predicted vertical and lateral intent or movement for every object in a list of two elements. Intent should be one of these values:
            1.1 Lateral (Sideways) Intent Options (has to be from these two options):",
                  "   - 'goes to the left'",
                  "   - 'goes to the right'\n",
        
                  "1.2. Vertical Intent Options:",
                  "   - 'moves away from ego vehicle'",
                  "   - 'moves towards ego vehicle'",
                  "   - 'stationary'\n",
        
        2. Reason for this prediction.
        3. Bounding box of the object in [x1, x2, y1, y2] format. these should be with respect to orginal image dimensions.
        
        
        the format of output dictionary is (these are just placeholders) {"name_of_object": {"Intent": ["goes to the left/goes to the right", "moves away from ego vehicle/moves towards ego vehicle/stationary"], "Reason": "reason for this prediction", "Bounding_box": [x1, y1, x2, y2] }} for all objects and NOT a list
        The Intent field list should ALWAYS have two values: one for lateral and one for vertical. For example --- "Intent": ["predicted lateral intent", "predicted vertical intent"]
        Strictly output in valid JSON format. This JSON shoud NOT contain details from the scene graph such as the relationships or attributes, stick to the format mentioned above.
        """
        return intent_prompt
    def all_gen_prompt(self):
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

        3. Risk score for this prediction (Yes or No). Risk is defined as a hazardous scenario that poses danger to the ego vehicle.
        4. Bounding box of each object. these should be with respect to orginal image dimensions.
        5. Suggested action given the scene and risk score

        an example structure would look like this: dictionary of {"Risk": Yes/No, "Suggested_action": "suggested action for ego vehicle", "name_of_object": {"Intent": ["predicted lateral intent", "predicted vertical intent"], "Reason": "reason for this prediction", "Bounding_box": [x1, y1, x2, y2] }} for all objects and NOT a list
        The Intent field list should ALWAYS have two values: one for lateral and one for vertical. For example --- "Intent": ["predicted lateral intent", "predicted vertical intent"].
        To reiterate, the following keys MUST be present in the JSON: "Risk", "Suggested_action", "name_of_object": {"Intent": [a, b], "Bounding_box": [x1, y1, x2, y2]]
        Strictly output in valid JSON format. 
        """
        
        return all_gen_prompt
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
        schema = self.scene_graph_prompt if prompt_type == 'scene' else self.intent_prompt
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
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        imagenp = np.array(image)

        # Format conversation for LLaVA using the proper format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{self.scene_graph_prompt}\nRespond ONLY with the JSON output."},
                    {"type": "text", "text": "ONLY output valid JSON. DO NOT repeat the prompt. DO NOT include explanations."}

                ]
            }
        ]

        # Apply chat template to get properly formatted prompt
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        # Process inputs
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)

        # Generate output
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            # do_sample=False
        )

        # Decode output
        output_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
        new_annotation = output_text.strip()
        new_annotation = new_annotation.split("```json", 1)[-1].lstrip() if "```json" in new_annotation else new_annotation

        new_annotation = new_annotation.replace("```json", "").replace("```", "")
        # Extract JSON from the response
        # new_annotation = output_text
        # print(new_annotation)

        # Clean up response if it contains markdown code blocks
        if new_annotation.startswith("```json"):
            new_annotation = new_annotation[7:]  # Remove the first 7 characters
        if new_annotation.endswith("```"):
            new_annotation = new_annotation[:-3]

        try:
            new_annotation = self.extract_and_fix_json(new_annotation, prompt_type="scene")
            # scene_graph_json = json.loads(new_annotation)
            return new_annotation
        except json.JSONDecodeError as e:
            scene_graph_json = {}
            print("Failed to convert string to dictionary:", e)
        return scene_graph_json

    def process_image_for_intent(self, image_url, scene_graph):
        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            imagenp = np.array(image)

            # Format conversation for LLaVA using the proper format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"{self.intent_prompt}\n This is the scene graph : {scene_graph}.\nThe Intent JSON is:."},
                        {"type": "text", "text": "ONLY output valid JSON. DO NOT repeat the prompt. DO NOT include explanations. DO NOT OUTPUT THE SCENE GRAPH AGAIN. Make sure the Intent List contains two elements"}

                    ]
                }
            ]

            # Apply chat template to get properly formatted prompt
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

            # Process inputs
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)

            # Generate output
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                # do_sample=False
            )

            # Decode output
            output_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
            new_annotation = output_text.strip()
            new_annotation = new_annotation.split("```json", 1)[-1].lstrip() if "```json" in new_annotation else new_annotation

            new_annotation = new_annotation.replace("```json", "").replace("```", "")
            # Extract JSON from the response
            # new_annotation = output_text

            # Clean up response if it contains markdown code blocks
            if new_annotation.startswith("```json"):
                new_annotation = new_annotation[7:]  # Remove the first 7 characters
            if new_annotation.endswith("```"):
                new_annotation = new_annotation[:-3]

        except requests.exceptions.RequestException as e:
            print(f"Error downloading image for {image_url}: {e}")
            return {}
        except Exception as e:
            print(f"Error processing frame {image_url}: {e}")
            return {}

        # print(new_annotation)
        try:
            new_annotation = self.extract_and_fix_json(new_annotation, prompt_type="intent")
            # intent_json = json.loads(new_annotation)
            return new_annotation
        except json.JSONDecodeError as e:
            intent_json = {}
            print("Failed to convert string to dictionary:", e)
        return intent_json

    def process_image_one_pass(self, image_url):
        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            imagenp = np.array(image)

            # Format conversation for LLaVA using the proper format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f'{self.all_gen_prompt} Output all the necessary keys and values , e.g : {{"Risk": Yes/No, "Suggested_action": "suggested action for ego vehicle", "name_of_object": {{"Intent": ["predicted lateral intent", "predicted vertical intent"], "Reason": "reason for this prediction", "Bounding_box": [x1, y1, x2, y2] }}}}\n'},
                        # {"type": "text", "text": "ONLY output valid JSON. DO NOT repeat the prompt. DO NOT include explanations. Make sure the Intent List contains two elements"}

                    ]
                }
            ]

            # Apply chat template to get properly formatted prompt
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

            # Process inputs
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)

            # Generate output
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                # do_sample=False
            )

            # Decode output
            output_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
            new_annotation = output_text.strip()
            new_annotation = new_annotation.split("```json", 1)[-1].lstrip() if "```json" in new_annotation else new_annotation

            new_annotation = new_annotation.replace("```json", "").replace("```", "")
            # Extract JSON from the response
            # new_annotation = output_text

            # Clean up response if it contains markdown code blocks
            if new_annotation.startswith("```json"):
                new_annotation = new_annotation[7:]  # Remove the first 7 characters
            if new_annotation.endswith("```"):
                new_annotation = new_annotation[:-3]

        except requests.exceptions.RequestException as e:
            print(f"Error downloading image for {image_url}: {e}")
            return {}
        except Exception as e:
            print(f"Error processing frame {image_url}: {e}")
            return {}

        # print(new_annotation)
        try:
            new_annotation = self.extract_and_fix_json(new_annotation, prompt_type="intent")
            # intent_json = json.loads(new_annotation)
            return new_annotation
        except json.JSONDecodeError as e:
            intent_json = {}
            print("Failed to convert string to dictionary:", e)
        return intent_json

    def run_inference(self, limit=None, overwrite=True, raw = False):
        """Run inference on dataset with concurrent API calls"""
        data = self.load_data()
        out_dir = os.path.join(os.path.dirname(self.dataset_path), "outputs", "llava")
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
        processed = 0
        for frame_id, frame_data in tqdm(items, total = len(items)):
            try:
              image_url = frame_data['image_path']
              if raw:
                rw = self.process_image_one_pass(image_url)
                # print(f'Raw JSON: {json.dumps(rw, indent = 4)}')
                all_raw[frame_id] = rw
              else:
                sg = self.process_image_for_scene_graph(image_url)
                intent = self.process_image_for_intent(image_url, sg)
                # print(f'Scene graph: {json.dumps(scene_graph, indent = 4)}')
                # print(f'Intent JSON: {json.dumps(intent_json, indent = 4)}')
                all_sg[frame_id]      = sg
                all_intent[frame_id]  = intent
              processed       += 1

                # save after each batch

            except Exception as e:
                print(f'Error in frame {frame_id}: {e}')
                # raise
                continue
            if not processed % 10:
                print(f'Processed {processed} images!')
                if raw:
                  with open(raw_path, "w") as f: json.dump(all_raw, f)
                else:
                  with open(sg_path, "w") as f: json.dump(all_sg, f)
                  with open(intent_path, "w") as f: json.dump(all_intent, f)
        if raw:
          with open(raw_path, "w") as f: json.dump(all_raw, f)
          return all_raw, None
        else:
          with open(sg_path, "w") as f: json.dump(all_sg, f)
          with open(intent_path, "w") as f: json.dump(all_intent, f)

      

        print('Done with all images!')
        return all_sg, all_intent

# Example usage:
if __name__ == "__main__":
    llava_inference = LlavaInference(dataset_path = '<dataset-path>')
    scene_graph, intent_json = llava_inference.run_inference(overwrite = True, raw = True, limit = None)
    print(scene_graph)
