# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import logging
import os
import re
import tempfile
import time
from PIL import Image
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Any, List, Optional
from dotenv import load_dotenv
import io
import numpy as np
import openai
import tiktoken
import httpx
from volcenginesdkarkruntime import Ark
from gui_rewalk.env.prompts import SYS_PROMPT_IN_SCREENSHOT_OUT_CODE, SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION, \
    SYS_PROMPT_IN_A11Y_OUT_CODE, SYS_PROMPT_IN_A11Y_OUT_ACTION, \
    SYS_PROMPT_IN_BOTH_OUT_CODE, SYS_PROMPT_IN_BOTH_OUT_ACTION, SYS_PROMPT_IN_SCREENSHOT_OUT_OCR_GUIAGENTACTION, \
    SYS_PROMPT_IN_SOM_OUT_TAG, SYS_PROMPT_IN_SCREENSHOT_OUT_GUIAGENTACTION
from gui_rewalk.src.config.config import CLICK_ACTION_TYPES

logger = logging.getLogger("desktopenv.agent")

ERROR_CALLING_LLM = 'Error calling LLM'
pure_text_settings = ['a11y_tree']

attributes_ns_ubuntu = "https://accessibility.windows.example.org/ns/attributes"
attributes_ns_windows = "https://accessibility.windows.example.org/ns/attributes"
state_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/state"
state_ns_windows = "https://accessibility.windows.example.org/ns/state"
component_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/component"
component_ns_windows = "https://accessibility.windows.example.org/ns/component"
value_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/value"
value_ns_windows = "https://accessibility.windows.example.org/ns/value"
class_ns_windows = "https://accessibility.windows.example.org/ns/class"
# More namespaces defined in OSWorld, please check desktop_env/server/main.py


def observation_encode_image(image_content):
    return base64.b64encode(image_content).decode('utf-8')
# Function to encode the image
def encode_image(image_content):
    return base64.b64encode(array_to_jpeg_bytes(image_content)).decode('utf-8')

def array_to_jpeg_bytes(image) -> bytes:
    """Converts a numpy array into a byte string for a JPEG image."""
    if isinstance(image, bytes):
        return image
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        return image_to_jpeg_bytes(image)
    else:
        raise ValueError("Invalid image type, must be numpy array or bytes")

def image_to_jpeg_bytes(image: Image.Image) -> bytes:
    in_mem_file = io.BytesIO()
    # Check image mode, convert RGBA to RGB if necessary
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else image.convert('RGBA').split()[-1])
        image = background
    # Reset file pointer to start
    image.save(in_mem_file, format='JPEG')
    in_mem_file.seek(0)
    img_bytes = in_mem_file.read()
    return img_bytes

def encoded_img_to_pil_img(data_str):
    base64_str = data_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    return image


def save_to_tmp_img_file(data_str):
    base64_str = data_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    tmp_img_path = os.path.join(tempfile.mkdtemp(), "tmp_img.png")
    image.save(tmp_img_path)

    return tmp_img_path


def linearize_accessibility_tree(accessibility_tree, platform="ubuntu"):

    if platform == "ubuntu":
        _attributes_ns = attributes_ns_ubuntu
        _state_ns = state_ns_ubuntu
        _component_ns = component_ns_ubuntu
        _value_ns = value_ns_ubuntu
    elif platform == "windows":
        _attributes_ns = attributes_ns_windows
        _state_ns = state_ns_windows
        _component_ns = component_ns_windows
        _value_ns = value_ns_windows
    else:
        raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")

    filtered_nodes = filter_nodes(ET.fromstring(accessibility_tree), platform)
    linearized_accessibility_tree = ["tag\tname\ttext\tclass\tdescription\tposition (top-left x&y)\tsize (w&h)"]

    # Linearize the accessibility tree nodes into a table format
    for node in filtered_nodes:
        if node.text:
            text = (
                node.text if '"' not in node.text \
                    else '"{:}"'.format(node.text.replace('"', '""'))
            )

        elif node.get("{{{:}}}class".format(class_ns_windows), "").endswith("EditWrapper") \
                and node.get("{{{:}}}value".format(_value_ns)):
            node_text = node.get("{{{:}}}value".format(_value_ns), "")
            text = (node_text if '"' not in node_text \
                        else '"{:}"'.format(node_text.replace('"', '""'))
                    )
        else:
            text = '""'

        linearized_accessibility_tree.append(
            "{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}".format(
                node.tag, node.get("name", ""),
                text,
                node.get("{{{:}}}class".format(_attributes_ns), "") if platform == "ubuntu" else node.get("{{{:}}}class".format(class_ns_windows), ""),
                node.get("{{{:}}}description".format(_attributes_ns), ""),
                node.get('{{{:}}}screencoord'.format(_component_ns), ""),
                node.get('{{{:}}}size'.format(_component_ns), "")
            )
        )

    return "\n".join(linearized_accessibility_tree)


def tag_screenshot(screenshot, accessibility_tree, platform="ubuntu"):
    nodes = filter_nodes(ET.fromstring(accessibility_tree), platform=platform, check_image=True)
    # Make tag screenshot
    marks, drew_nodes, element_list, tagged_screenshot = draw_bounding_boxes(nodes, screenshot)

    return marks, drew_nodes, tagged_screenshot, element_list


def parse_actions_from_string(input_string):
    if input_string.strip() in ['WAIT', 'DONE', 'FAIL']:
        return [input_string.strip()]
    # Search for a JSON string within the input string
    actions = []
    matches = re.findall(r'```json\s+(.*?)\s+```', input_string, re.DOTALL)
    if matches:
        # Assuming there's only one match, parse the JSON string into a dictionary
        try:
            for match in matches:
                action_dict = json.loads(match)
                actions.append(action_dict)
            return actions
        except json.JSONDecodeError as e:
            return f"Failed to parse JSON: {e}"
    else:
        matches = re.findall(r'```\s+(.*?)\s+```', input_string, re.DOTALL)
        if matches:
            # Assuming there's only one match, parse the JSON string into a dictionary
            try:
                for match in matches:
                    action_dict = json.loads(match)
                    actions.append(action_dict)
                return actions
            except json.JSONDecodeError as e:
                return f"Failed to parse JSON: {e}"
        else:
            try:
                action_dict = json.loads(input_string)
                return [action_dict]
            except json.JSONDecodeError:
                raise ValueError("Invalid response format: " + input_string)


def parse_code_from_string(input_string):
    input_string = "\n".join([line.strip() for line in input_string.split(';') if line.strip()])
    if input_string.strip() in ['WAIT', 'DONE', 'FAIL']:
        return [input_string.strip()]

    # This regular expression will match both ```code``` and ```python code```
    # and capture the `code` part. It uses a non-greedy match for the content inside.
    pattern = r"```(?:\w+\s+)?(.*?)```"
    # Find all non-overlapping matches in the string
    matches = re.findall(pattern, input_string, re.DOTALL)

    # The regex above captures the content inside the triple backticks.
    # The `re.DOTALL` flag allows the dot `.` to match newline characters as well,
    # so the code inside backticks can span multiple lines.

    # matches now contains all the captured code snippets

    codes = []

    for match in matches:
        match = match.strip()
        commands = ['WAIT', 'DONE', 'FAIL']  # fixme: updates this part when we have more commands

        if match in commands:
            codes.append(match.strip())
        elif match.split('\n')[-1] in commands:
            if len(match.split('\n')) > 1:
                codes.append("\n".join(match.split('\n')[:-1]))
            codes.append(match.split('\n')[-1])
        else:
            codes.append(match)

    return codes

def parse_action_from_string(input_string):
    return input_string


def parse_code_from_som_string(input_string, masks):
    # parse the output string by masks
    tag_vars = ""
    for i, mask in enumerate(masks):
        x, y, w, h = mask
        tag_vars += "tag_" + str(i + 1) + "=" + "({}, {})".format(int(x + w // 2), int(y + h // 2))
        tag_vars += "\n"

    actions = parse_code_from_string(input_string)

    for i, action in enumerate(actions):
        if action.strip() in ['WAIT', 'DONE', 'FAIL']:
            pass
        else:
            action = tag_vars + action
            actions[i] = action

    return actions


def trim_accessibility_tree(linearized_accessibility_tree, max_tokens):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(linearized_accessibility_tree)
    if len(tokens) > max_tokens:
        linearized_accessibility_tree = enc.decode(tokens[:max_tokens])
        linearized_accessibility_tree += "[...]\n"
    return linearized_accessibility_tree


class GUIGenAgent:
    def __init__(
            self,
            completion_model="",
            platform="ubuntu",
            model=None,
            model_version=None,
            max_tokens=1500,
            top_p=0.9,
            temperature=0.5,
            action_space="gen_data",
            observation_type="screenshot_a11y_tree",
            # observation_type can be in ["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"]
            max_trajectory_length=3,
            a11y_tree_max_tokens=10000,
            max_retry=3,
            enable_ocr=False,
            enable_thinking=False,
            use_ark=False
    ):
        self.platform = platform
        self.model = model
        self.model_version = model_version
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.enable_ocr = enable_ocr
        self.a11y_tree_max_tokens = a11y_tree_max_tokens
        self.max_retry = max_retry
        self.enable_thinking = enable_thinking
        self.use_ark = use_ark

        self.thoughts = []
        self.actions = []
        self.observations = []
        
        if self.use_ark:
            self.model_client = Ark(
                api_key=os.environ.get("API_KEY"),
                base_url=os.environ.get("API_BASE_URL"),
                http_client=httpx.Client(trust_env=False),
            )
        else:
            self.model_client = openai.OpenAI(
                api_key=os.environ.get("API_KEY"),
                base_url=os.environ.get("API_BASE_URL"),
            )
        

        if observation_type == "screenshot":
            if action_space == "computer_13":
                self.system_message = SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION
            elif action_space == "pyautogui":
                self.system_message = SYS_PROMPT_IN_SCREENSHOT_OUT_CODE
            elif action_space == 'gen_data' and self.enable_ocr:
                self.system_message = SYS_PROMPT_IN_SCREENSHOT_OUT_OCR_GUIAGENTACTION.format(max_trajectory_length=self.max_trajectory_length)
            elif action_space == 'gen_data' and not self.enable_ocr:
                self.system_message = SYS_PROMPT_IN_SCREENSHOT_OUT_GUIAGENTACTION
            else:
                raise ValueError("Invalid action space: " + action_space)
        elif observation_type == "a11y_tree":
            if action_space == "computer_13":
                self.system_message = SYS_PROMPT_IN_A11Y_OUT_ACTION
            elif action_space == "pyautogui":
                self.system_message = SYS_PROMPT_IN_A11Y_OUT_CODE
            else:
                raise ValueError("Invalid action space: " + action_space)
        elif observation_type == "screenshot_a11y_tree":
            if action_space == "computer_13":
                self.system_message = SYS_PROMPT_IN_BOTH_OUT_ACTION
            elif action_space == "pyautogui":
                self.system_message = SYS_PROMPT_IN_BOTH_OUT_CODE
            elif action_space == 'gen_data':
                self.system_message = SYS_PROMPT_IN_SCREENSHOT_OUT_GUIAGENTACTION
            else:
                raise ValueError("Invalid action space: " + action_space)
        elif observation_type == "som":
            if action_space == "computer_13":
                raise ValueError("Invalid action space: " + action_space)
            elif action_space == "pyautogui":
                self.system_message = SYS_PROMPT_IN_SOM_OUT_TAG
            else:
                raise ValueError("Invalid action space: " + action_space)
        else:
            raise ValueError("Invalid experiment type: " + observation_type)

    def predict(self, instruction, obs, ocr, idx, directory) -> List:
        """
        Predict the next action(s) based on the current observation.
        """
        system_message = self.system_message + "\nYou are asked to complete the following task: {}".format(instruction)
        
        # Prepare the payload for the API call
        messages = []
        masks = None

        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                },
            ]
        })

        # Append trajectory
        assert len(self.observations) == len(self.actions) and len(self.actions) == len(self.thoughts) \
            , "The number of observations and actions should be the same."

        if len(self.observations) > self.max_trajectory_length:
            if self.max_trajectory_length == 0:
                _observations = []
                _actions = []
                _thoughts = []
            else:
                _observations = self.observations[-self.max_trajectory_length:]
                _actions = self.actions[-self.max_trajectory_length:]
                _thoughts = self.thoughts[-self.max_trajectory_length:]
        else:
            _observations = self.observations
            _actions = self.actions
            _thoughts = self.thoughts

        for previous_obs, previous_action, previous_thought in zip(_observations, _actions, _thoughts):

            if self.observation_type == "screenshot_a11y_tree":
                _screenshot = previous_obs["screenshot"]
                _linearized_accessibility_tree = previous_obs["accessibility_tree"]

                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Given the screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                                _linearized_accessibility_tree)
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_screenshot}",
                                "detail": "high"
                            }
                        }
                    ]
                })
            elif self.observation_type in ["som"]:
                _screenshot = previous_obs["screenshot"]

                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Given the tagged screenshot as below. What's the next step that you will do to help with the task?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_screenshot}",
                                "detail": "high"
                            }
                        }
                    ]
                })
            elif self.observation_type == "screenshot":
                _screenshot = previous_obs["screenshot"]

                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Given the screenshot as below. What's the next step that you will do to help with the task?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{_screenshot}",
                                "detail": "high"
                            }
                        }
                    ]
                })
            elif self.observation_type == "a11y_tree":
                _linearized_accessibility_tree = previous_obs["accessibility_tree"]

                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Given the info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                                _linearized_accessibility_tree)
                        }
                    ]
                })
            else:
                raise ValueError("Invalid observation_type type: " + self.observation_type)  

            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"Previous Action: {previous_action}\n\nPrevious Thought: {previous_thought.strip() if len(previous_thought) > 0 else 'No valid action'}"
                    },
                ]
            })


        if self.observation_type in ["screenshot", "screenshot_a11y_tree"]:
            if self.enable_ocr:
                ui_elements, logits = ocr.detect_gui(obs["screenshot"])
                user_prompt = f"bbox id num: {len(ui_elements)}, return the id from 0 to {len(ui_elements) - 1}"
                image_detect_array = ocr.draw_bbox(obs["screenshot"], ui_elements, logits)    
                base64_image = encode_image(image_detect_array)
                # save ocr img
                # result_image = Image.fromarray(image_detect_array)
                # result_image.save(os.path.join(directory, f"{idx}_before_ocr.png"))
            else:
                ui_elements = None
                base64_image = observation_encode_image(obs["screenshot"])
            linearized_accessibility_tree = linearize_accessibility_tree(accessibility_tree=obs["accessibility_tree"],
                                                                         platform=self.platform) if self.observation_type == "screenshot_a11y_tree" else None
            logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(linearized_accessibility_tree,
                                                                        self.a11y_tree_max_tokens)

            if self.observation_type == "screenshot_a11y_tree":
                self.observations.append({
                    "screenshot": base64_image,
                    "accessibility_tree": linearized_accessibility_tree
                })
            else:
                self.observations.append({
                    "screenshot": base64_image,
                    "accessibility_tree": None
                })

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Given the screenshot as below, {user_prompt}. What's the next step that you will do to help with the task?"
                        if self.observation_type == "screenshot"
                        else "Given the screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                            linearized_accessibility_tree)
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            })
        elif self.observation_type == "a11y_tree":
            linearized_accessibility_tree = linearize_accessibility_tree(accessibility_tree=obs["accessibility_tree"],
                                                                         platform=self.platform)
            logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(linearized_accessibility_tree,
                                                                        self.a11y_tree_max_tokens)

            self.observations.append({
                "screenshot": None,
                "accessibility_tree": linearized_accessibility_tree
            })

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Given the info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                            linearized_accessibility_tree)
                    }
                ]
            })
        elif self.observation_type == "som":
            # Add som to the screenshot
            masks, drew_nodes, tagged_screenshot, linearized_accessibility_tree = tag_screenshot(obs["screenshot"], obs[
                "accessibility_tree"], self.platform)
            base64_image = encode_image(tagged_screenshot)
            logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(linearized_accessibility_tree,
                                                                        self.a11y_tree_max_tokens)

            self.observations.append({
                "screenshot": base64_image,
                "accessibility_tree": linearized_accessibility_tree
            })

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Given the tagged screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                            linearized_accessibility_tree)
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            })
        else:
            raise ValueError("Invalid observation_type type: " + self.observation_type) 
        
        prompt_token_total = 0
        completion_token_total = 0
        counter = 1
        while counter <= self.max_retry:
            try:
                response, prompt_token, completion_token = self.get_api_info(messages, self.model_version)
                prompt_token_total += prompt_token
                completion_token_total += completion_token
                    
                logger.info("RESPONSE: %s", response)
            
                result, thoughts, summary_problem = self.parse_action_from_ocr_completion(response, ui_elements)
                if thoughts is None or result is None:
                    counter += 1
                    continue
            
                actions = self.parse_actions(result, masks)
                # print(f'actions: {actions}')
                self.thoughts.append(thoughts)
                return thoughts, actions, summary_problem, prompt_token_total, completion_token_total, counter

            except ValueError as e:
                print("Failed to parse action from response", e)
                actions = None
                counter += 1
        return None, None, None, prompt_token_total, completion_token_total, counter

    def predict_mm(
            self, text_prompt: str, images: list[np.ndarray]
        ) -> tuple[str, Optional[bool], Any]:
        """Multi-modal inference interface, support text + image input"""

        content = [{"type": "text", "text": text_prompt}]
       
        for image in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(image)}"
                },
            })

        messages = [
            {
                "role": "user",
                "content": content,
            },
        ]

        counter = 1
        prompt_token_total = 0
        completion_token_total = 0
        wait_seconds = 20


        while counter <= self.max_retry:
            try:
                response, prompt_token, completion_token = self.get_api_info(messages, self.model_version)
                prompt_token_total += prompt_token
                completion_token_total += completion_token
                return response, prompt_token_total, completion_token_total, counter
            except Exception as e:  # pylint: disable=broad-exception-caught
                counter += 1
                print(f'Error calling LLM API, will retry in {wait_seconds} seconds')
                print(e)
                if counter > 0:
                    time.sleep(wait_seconds)
                wait_seconds *= 2
        return ERROR_CALLING_LLM, None, None, counter
        
    def predict_random_rewalk(
            self, text_prompt: str, images: list[np.ndarray]
        ) -> tuple[str, Optional[bool], Any]:
        """多模态推理接口，支持文本+图像输入"""


        content = [{"type": "text", "text": text_prompt}]
       

        for image in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(image)}"
                },
            })

        messages = [
            {
                "role": "user",
                "content": content,
            },
        ]

        counter = 1
        prompt_token_total = 0
        completion_token_total = 0
        wait_seconds = 20

        while counter <= 1:
            try:
                response, prompt_token, completion_token = self.get_api_info(messages, self.model_version)
                prompt_token_total += prompt_token
                completion_token_total += completion_token
                action_json = self.parse_json(response)
                ocr_items = list(action_json.items())
                return ocr_items, prompt_token_total, completion_token_total, counter
            except Exception as e:  # pylint: disable=broad-exception-caught
                counter += 1
                print(f'Error calling LLM API, will retry in {wait_seconds} seconds')
                print(e)
                if counter > 0:
                    time.sleep(wait_seconds)
                wait_seconds *= 2
        return ERROR_CALLING_LLM, None, None, counter
    

    def get_api_info(self, messages, model):
        completion = self.model_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
        )

        if isinstance(completion.usage, dict):
            total_tokens = completion.usage.get('total_tokens', 'N/A')
            prompt_tokens = completion.usage.get('prompt_tokens', 'N/A')
            completion_tokens = completion.usage.get('completion_tokens', 'N/A')
        else:
            total_tokens = getattr(completion.usage, 'total_tokens', 'N/A')
            prompt_tokens = getattr(completion.usage, 'prompt_tokens', 'N/A')
            completion_tokens = getattr(completion.usage, 'completion_tokens', 'N/A')
        # logger.info("RESPONSE: %s", completion.choices[0].message.content)
        # logger.info("total_tokens: %s, prompt_tokens: %s, completion_tokens: %s", total_tokens, prompt_tokens, completion_tokens)
        # print(f'total_tokens: {completion.usage.total_tokens}, prompt_token: {completion.usage.prompt_tokens}, completion_tokens: {completion.usage.completion_tokens}')
        if completion.choices and len(completion.choices) > 0:
            return (
                completion.choices[0].message.content,
                prompt_tokens,
                completion_tokens,
            )
        else:
            print('No choices returned from LLM API')
        
    def parse_action_from_ocr_completion(self, response, ui_elements=None):
        # 提取 thoughts 内容
        thoughts_match = re.search(r'"thoughts": "(.*?)"', response, re.DOTALL)
        thoughts = thoughts_match.group(1) if thoughts_match else ""
        summary_problem_match = re.search(r'"summary_problem": "(.*?)"', response, re.DOTALL)
        summary_problem = summary_problem_match.group(1) if summary_problem_match else ""


        result_match = re.search(r'"result": "(.*?)"', response, re.DOTALL)
        result = result_match.group(1) if result_match else ""
        if self.enable_ocr:
            ocr_match = re.search(r'"ocr": "(.*?)"', response, re.DOTALL)
            ocr = ocr_match.group(1) if ocr_match else ""
            if ocr.isdigit():
                box_id = int(ocr)
                if box_id >= len(ui_elements):
                    return thoughts, result
                xyxy = ui_elements[box_id]
                x, y = int((xyxy[0] + xyxy[2]) // 2), int((xyxy[1] + xyxy[3]) // 2)
                match = re.search(r"(\w+)\(", result)
                action_type = match.group(1).upper()
                if action_type in CLICK_ACTION_TYPES and action_type != 'DRAG':
                    result = f"{action_type}(point='<point>{x} {y}</point>')"
        return result, thoughts, summary_problem
    
    def parse_actions(self, response: str, masks=None):

        if self.observation_type in ["screenshot", "a11y_tree", "screenshot_a11y_tree"]:
            # parse from the response
            if self.action_space == "computer_13":
                actions = parse_actions_from_string(response)
            elif self.action_space == "pyautogui":
                actions = parse_code_from_string(response)
            elif self.action_space == "gen_data":
                actions = parse_action_from_string(response)
            else:
                raise ValueError("Invalid action space: " + self.action_space)

            self.actions.append(actions)

            return actions
        elif self.observation_type in ["som"]:
            # parse from the response
            if self.action_space == "computer_13":
                raise ValueError("Invalid action space: " + self.action_space)
            elif self.action_space == "pyautogui":
                actions = parse_code_from_som_string(response, masks)
            else:
                raise ValueError("Invalid action space: " + self.action_space)

            self.actions.append(actions)

            return actions
        
    def parse_json(self, response, fields=None):
        """
        Extract JSON from LLM output, optionally preserving only specific fields (e.g., task and app).

        Args:
            llm_output (str or tuple): String or tuple returned by LLM.
            fields (list[str], optional): List of fields to extract, such as ['task', 'app'].

        Returns:
            dict or None: Extraction result, which may be the complete JSON or a subset of fields.
        """

        if isinstance(response, tuple):
            response = response[0]
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if not match:

            match = re.search(r"\{.*?\}", response, re.DOTALL)

        if match:
            json_str = match.group(1)
            try:
                data = json.loads(json_str)
                if fields:
                    return {key: data.get(key, "") for key in fields}
                return data
            except json.JSONDecodeError as e:
                print("Failed to parse JSON:", e, response)
                return None
        else:
            print("No JSON block found.", response)
            return None

    def reset(self, _logger=None):
        global logger
        logger = _logger if _logger is not None else logging.getLogger("desktopenv.agent")

        self.thoughts = []
        self.actions = []
        self.observations = []
