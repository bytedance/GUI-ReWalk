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

import json
from typing import Dict, Any, Optional, Tuple, Union

def check_task_status(
    agent,
    screenshot,
    action_history: Optional[str] = None,
    goal: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    检查当前任务状态，判断任务是否完成
    
    参数:
        agent: AI代理对象
        screenshot: 当前屏幕截图
        action_history: 历史动作记录（可选）
        goal: 任务目标（可选）
        
    返回:
        Tuple[str, Dict[str, Any]]: (状态码, 状态详细信息)
        状态码可能为: 'complete', 'infeasible', 'continue', 'error'
    """
    try:
        # 构建检查prompt
        check_prompt = PROMPT_PREFIX.format(os_type=OS_TYPE) + "\n\n"
        
        if goal:
            check_prompt += f"Current Goal: {goal}\n\n"
        
        if action_history:
            check_prompt += f"Action History:\n{action_history}\n\n"
        
        check_prompt += (
            "Based on the current screenshot and action history, "
            "determine if the task has been completed or if it's infeasible. "
            "Respond with one of these JSON formats:\n"
            '- Task completed: {"action_type": "status", "goal_status": "complete"}\n'
            '- Task infeasible: {"action_type": "status", "goal_status": "infeasible"}\n'
            '- Continue working: {"action_type": "status", "goal_status": "continue"}\n'
            "Only respond with the JSON format, no additional text."
        )
        
        # 使用agent的LLM进行判断
        if hasattr(agent, 'model'):
            response_text, prompt_token, completion_token, retry_counter = agent.predict_mm(check_prompt, [screenshot])
            
            # 尝试解析JSON响应
            try:
                # 清理响应文本，提取JSON部分
                response_text = response_text.strip()
                if response_text.startswith('```'):
                    # 移除可能的markdown代码块标记
                    lines = response_text.split('\n')
                    json_lines = []
                    in_json = False
                    for line in lines:
                        if line.strip().startswith('{'):
                            in_json = True
                        if in_json:
                            json_lines.append(line)
                        if line.strip().endswith('}'):
                            break
                    response_text = '\n'.join(json_lines)
                
                # 查找JSON对象
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    status_response = json.loads(json_str)
                    
                    goal_status = status_response.get('goal_status', 'continue')
                    
                    status_info = {
                        'raw_response': response_text,
                        'parsed_response': status_response,
                        'goal_status': goal_status,
                        'status_prompt_token': prompt_token,
                        'status_completion_token': completion_token,
                        'status_retry_counter': retry_counter,
                        'check_successful': True
                    }
                    
                    print(f"任务状态检查结果: {goal_status}")
                    return goal_status, status_info
                else:
                    raise ValueError("No valid JSON found in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                print(f"解析状态检查响应失败: {e}")
                print(f"原始响应: {response_text}")
                
                # 回退方案：基于关键词判断
                response_lower = response_text.lower()
                if 'complete' in response_lower:
                    goal_status = 'complete'
                elif 'infeasible' in response_lower:
                    goal_status = 'infeasible'
                else:
                    goal_status = 'continue'
                
                status_info = {
                    'raw_response': response_text,
                    'parsed_response': None,
                    'goal_status': goal_status,
                    'check_successful': False,
                    'error': str(e)
                }
                
                return goal_status, status_info
        else:
            print("Agent没有LLM能力，无法进行状态检查")
            status_info = {
                'raw_response': 'No LLM available',
                'parsed_response': None,
                'goal_status': 'continue',
                'check_successful': False,
                'error': 'No LLM available'
            }
            return 'continue', status_info
            
    except Exception as e:
        print(f"状态检查过程中发生错误: {e}")
        status_info = {
            'raw_response': None,
            'parsed_response': None,
            'goal_status': 'error',
            'check_successful': False,
            'error': str(e)
        }
        return 'error', status_info


def format_action_history(trajectory: list) -> str:
    """
    格式化动作历史为可读的字符串
    
    参数:
        trajectory: 轨迹数据列表
        
    返回:
        str: 格式化的动作历史
    """
    if not trajectory:
        return "No previous actions."
    
    history_lines = []
    for i, step in enumerate(trajectory, 1):
        action_type = step.get('action_type', 'unknown')
        action_json = step.get('action_json', '{}')
        
        try:
            action_data = json.loads(action_json) if isinstance(action_json, str) else action_json
            if action_type == 'CLICK':
                index = action_data.get('index', 'unknown')
                history_lines.append(f"Step {i}: Clicked on element {index}")
            elif action_type == 'TYPE':
                text = action_data.get('text', 'unknown')
                index = action_data.get('index', 'unknown')
                history_lines.append(f"Step {i}: Typed '{text}' into element {index}")
            else:
                history_lines.append(f"Step {i}: {action_type} action")
        except:
            history_lines.append(f"Step {i}: {action_type} action")
    
    return '\n'.join(history_lines) 

def format_history_instruction(trajectory: list) -> str:
    """
    格式化历史指令为可读的字符串
    """
    if not trajectory:
        return "No previous actions."
    
    history_lines = []
    for i, step in enumerate(trajectory, 1):
        instruction = step.get('high_level_instruction', 'unknown')
        
        try:
            history_lines.append(f"Step {i}: {instruction}")
        except:
            history_lines.append(f"Step {i}: {instruction}")
    
    return '\n'.join(history_lines) 


import os
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image, ImageDraw
import concurrent.futures

# 导入先前开发的模块
from ..config.config import OS_TYPE, TOTAL_ACTION_TYPES
from ..config.prompt import REVERSE_INFERENCE_TEMPLATE, PROMPT_PREFIX
from ..utils.utils import extract_action_from_json, format_action_for_prompt


# 设置常量

PROMPT_TEMPLATE = """
You are an expert at envisioning specific tasks corresponding to changes in mobile screenshots. I will provide you with the following:

1. The type of action currently being executed. The type of action currently being executed, which can be one of five types: CLICK, SCROLL, TYPE, PRESS_BACK, and LONG_PRESS. If the action is TYPE, an additional value representing the input will be provided. If the action is SCROLL, an additional scroll direction will be provided.

2. Screenshots of the interface before and after the current action is performed. If the action is CLICK, the pre-action screenshot will include a red bbox highlighting the element being interacted with (if applicable). Pay particular attention to the content of the element corresponding to the red bbox.

3. The name of the app where the current screenshot is located.

4. The action history of the current task.

5. The history instruction of the current task.

Your task is to envision a specific task based on the current action and the corresponding changes in screenshots. The output should include three parts:

1. Sub-Instruction: Based on the interface change caused by the current action, generate a corresponding natural language instruction for the current action. The instruction should be concise, clear, and executable. It must include specific details critical to the operation, such as file names, times, or other content as they appear in the screenshots. For example: "Scroll left to open the app drawer, displaying all installed applications on the devic", "Click the chat interface, allowing the user to view and participate in conversation", "Type the username 'Agent', preparing for the next step in logging into the account".

2. Analysis: Analyze the changes from the historical interface to the current interface and the operation instructions.

3. Purpose: Based on the historical and existing screenshots and operations, infer what I am doing and what kind of task I hope to complete.

4. High-Level-Instruction: Based on the analysis results, envision a reasonable and effective high-level task from the historical interface to the current interface. There are two types of high-level instructions: Task-Oriented: Complete a series of operations to achieve a specific goal. Question-Oriented: Perform a series of operations and derive an answer to a specific question.

Ensure that the High-Level-Instruction is executable by including all critical specifics, such as file names, relevant timings, or required details.

You ONLY need to return a dictionary formatted as follows:
{{
  "Sub-Instruction": "xxx",
  "Analysis": "xxx",
  "Purpose": "xxx",
  "High-Level-Instruction": "xxx",
}}

Current Action: {current_action}
App Name: {app_name}
Action History: {action_history}
History Instruction: {history_instruction}
RETURN ME THE DICTIONARY I ASKED FOR.
"""


def call_llm_api(
    prompt: str, 
    before_image_path: str, 
    after_image_path: str, 
    before_rect_path: str,
    expanded_region_path: str, 
    agent
) -> Dict[str, str]:
    """
    使用Android_world Agent进行推理

    参数:
        prompt: 提示文本
        before_image_path: 动作前的截图路径
        after_image_path: 动作后的截图路径
        agent: Android_world Agent实例

    返回:
        Dict: 包含推理结果的字典
    """
    # 打开图像并转换为numpy数组
    image_list = []
    before_image = np.array(Image.open(before_image_path))
    image_list.append(before_image)
    after_image = np.array(Image.open(after_image_path))
    image_list.append(after_image)
    if before_rect_path != "":
        before_rect = np.array(Image.open(before_rect_path))
        image_list.append(before_rect)
    if expanded_region_path != "":
        expanded_region = np.array(Image.open(expanded_region_path))
        image_list.append(expanded_region)

    # 使用Agent进行推理
    # print(f'随机游走反向推理...调用doubao api')

    reverse_response, prompt_token, completion_token, retry_counter = agent.predict_mm(prompt, image_list)
    reverse_json = agent.parse_json(reverse_response)
    # print(f'随机游走反向推理...返回api结果')
    reverse_json["reverse_prompt_token"] = prompt_token
    reverse_json["reverse_completion_token"] = completion_token
    reverse_json["reverse_retry_counter"] = retry_counter
    return reverse_json


def expand_and_crop_ocr_region(
    image, 
    bbox, 
    scale_factor=2.0, 
    min_size=None,  # 最小尺寸限制（像素），取较长边
    max_size=None   # 最大尺寸限制（像素），取较长边
):
    """
    扩展OCR检测框区域后裁剪
    
    参数:
        image_path (str): 原始图像路径
        bbox (tuple): OCR检测框坐标 (x1, y1, x2, y2)
        scale_factor (float): 区域扩大倍数，默认为2.0
    
    返回:
        numpy.ndarray: 扩大并裁剪后的图像区域
    """
    # 解析检测框坐标
    x1, y1, x2, y2 = bbox
    
    # 计算原始区域中心坐标
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # 计算原始区域宽度和高度
    original_width = x2 - x1 if x2 != x1 else 1
    original_height = y2 - y1 if y2 != y1 else 1 

    # 计算扩大后的区域宽度和高度
    new_width = original_width * scale_factor
    new_height = original_height * scale_factor

    # 保持宽高比的情况下应用最小/最大尺寸限制
    if min_size is not None or max_size is not None:
        # 计算原始宽高比
        aspect_ratio = original_width / original_height
        
        # 应用最小尺寸限制
        if min_size is not None:
            min_width = min_size if aspect_ratio >= 1 else int(min_size * aspect_ratio)
            min_height = min_size if aspect_ratio < 1 else int(min_size / aspect_ratio)
            new_width = max(new_width, min_width)
            new_height = max(new_height, min_height)
        
        # 应用最大尺寸限制
        if max_size is not None:
            max_width = max_size if aspect_ratio >= 1 else int(max_size * aspect_ratio)
            max_height = max_size if aspect_ratio < 1 else int(max_size / aspect_ratio)
            new_width = min(new_width, max_width)
            new_height = min(new_height, max_height)
    # === 尺寸限制逻辑结束 ===
    
    # 计算扩大后的区域坐标
    new_x1 = center_x - new_width / 2
    new_y1 = center_y - new_height / 2
    new_x2 = center_x + new_width / 2
    new_y2 = center_y + new_height / 2
    
    # 确保坐标在图像范围内
    height = image.height
    width = image.width
    new_x1 = max(0, int(new_x1))
    new_y1 = max(0, int(new_y1))
    new_x2 = min(width, int(new_x2))
    new_y2 = min(height, int(new_y2))
    
    # 裁剪扩大后的区域
    expanded_region = image.crop((new_x1, new_y1, new_x2, new_y2))

    
    return expanded_region

def draw_and_crop_ocr_region(
    image_path: str,
    bbox: Union[Dict[str, Any], List[float]],
    color: str = "red",
    width: int = 2,
):
    """在图片上绘制坐标点
    Args:
        image: PIL Image对象
        center: [x, y]坐标
        radius: 圆点半径
        color: 颜色
    """

    if bbox is None or bbox == {}:
        return "", ""
    
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    if isinstance(bbox, list):
        if len(bbox) == 2:
            x, y = bbox
            if isinstance(x, list) and isinstance(y, list):
                x1, y1 = x[0], x[1]
                x2, y2 = y[0], y[1]
                x_min, y_min = x1, y1
                x_max, y_max = x2, y2
                radius1 = radius2 = width + 3
                draw.ellipse([x1 - radius1, y1 - radius1, x1 + radius1, y1 + radius1], fill=color)
                draw.ellipse([x2 - radius2, y2 - radius2, x2 + radius2, y2 + radius2], fill=color)
            else:
                radius = width + 3
                x_min, y_min = x - 50, y - 50
                x_max, y_max = x + 50, y + 50
                # 绘制圆点
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
        elif len(bbox) == 4:
            x_min, y_min, x_max, y_max = bbox
            # 绘制矩形框
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)
    else:
        x_min, y_min, x_max, y_max = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]
        if x_max == x_min or y_max == y_min:
            x_min -= 50
            y_min -= 50
            x_max += 50
            y_max += 50

        # 绘制矩形框
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)

    # 扩展OCR检测框区域
    expanded_region = expand_and_crop_ocr_region(image, [x_min, y_min, x_max, y_max],
                                                 scale_factor=4.0, min_size=200, max_size=800)

    # 保存图片
    # save_path = image_path.replace("_before.png", f"_before_rect.png")
    expanded_region_path = image_path.replace("_before.png", f"_before_expand.png")
    # image.save(save_path)
    expanded_region.save(expanded_region_path)
    return "", expanded_region_path


def process_trajectory(
    trajectory_data: List[Dict], app_name: str, agent = None, screenshot_dir: Optional[str] = None,
):
    """
    处理轨迹数据，推理每一步的指令和目标

    参数:
        trajectory_data: 轨迹数据列表
        app_name: 应用名称
        output_file: 输出文件路径
        agent: AI代理对象（可选）
    """
    # 确保输出目录存在
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)

    step_data_list = []

    # 处理每一步
    for i in range(len(trajectory_data)):
        current_step = trajectory_data[i]

        # 获取屏幕截图路径
        screen_before_path = os.path.join(
            current_step.get("screen_before", "")
        )

        screen_after_path = os.path.join(
            current_step.get("screen_after", "")
        )

        # 确保截图路径存在
        if not screen_before_path or not screen_after_path:
            print(f"Missing screenshots for step {i+1}, skipping...")
            continue

        # 提取动作
        action_json = json.loads(current_step.get("action_json", {}))
        action = extract_action_from_json(action_json)
        action_text = format_action_for_prompt(action)
        
        # 可视化框选范围、点击区域
        if "parameters" in action_json:
            parameters = action_json["parameters"]
            if "x_min" in action_json["parameters"] and "y_min" in action_json["parameters"] and "x_max" in action_json["parameters"] and "y_max" in action_json["parameters"]:
                bbox = [parameters["x_min"], parameters["y_min"], parameters["x_max"], parameters["y_max"]]
            elif "x" in action_json["parameters"] and "y" in action_json["parameters"]:
                bbox = [parameters["x"], parameters["y"]]
            elif "x1" in action_json["parameters"] and "y1" in action_json["parameters"] and "x2" in action_json["parameters"] and "y2" in action_json["parameters"]:
                bbox = [[parameters["x1"], parameters["y1"]], [parameters["x2"], parameters["y2"]]]
            else:
                bbox = action_json.get("bbox", {})
        else:
            bbox = action_json.get("bbox", {})
            
        screen_before_rect_path, expanded_region_path = draw_and_crop_ocr_region(
            screen_before_path, bbox
        )

        # 准备提示
        prompt = REVERSE_INFERENCE_TEMPLATE.format(
            os_type=OS_TYPE,
            current_action=action_text,
            action_types=TOTAL_ACTION_TYPES,
        )

        step_data_list.append({
            'step': i,
            'prompt': prompt,
            'screen_before_rect_path': screen_before_rect_path,
            'expanded_region_path': expanded_region_path,
            'screen_after_path': screen_after_path,
            'action_text': action_text,
            'action_json': current_step.get("action_json", {}),
            'screen_before': screen_before_path,
            'screen_after': screen_after_path,
            'trajectory_slice': trajectory_data[:i],
            "forward_prompt_token": current_step.get("forward_prompt_token", ""),
            "forward_completion_token": current_step.get("forward_completion_token", ""),
            "forward_retry_counter": current_step.get("forward_retry_counter", 0),
            'current_step': current_step
        })
        
        # 并发处理推理和状态检查
    results = [None] * len(step_data_list)

    def process_step(step_data):
        i = step_data['step']
        print(f"Processing reverse inference step {i+1}/{len(step_data_list)}...")

        # 使用Agent进行推理
        inference_result = call_llm_api(
            step_data['prompt'],
            step_data['screen_before'],
            step_data['screen_after'],
            step_data['screen_before_rect_path'],
            step_data['expanded_region_path'],
            agent
        )

        # 进行状态检查
        status_check_result = {}
        if agent:
            try:
                # 构建动作历史（当前步骤之前的所有步骤）
                action_history = ""
                for j in range(i):
                    prev_step = trajectory_data[j]
                    prev_action_json = json.loads(prev_step.get("action_json", {}))
                    prev_action = extract_action_from_json(prev_action_json)
                    action_history += f"Step {j+1}: {format_action_for_prompt(prev_action)}\n"

                # 使用高级指令作为目标
                goal = inference_result.get("High-Level-Instruction", "")

                # 读取当前截图进行状态检查
                current_screenshot = np.array(Image.open(step_data['screen_after_path']))

                # 执行状态检查
                status, status_info = check_task_status(
                    agent=agent,
                    screenshot=current_screenshot,
                    action_history=action_history,
                    goal=goal
                )

                status_check_result = {
                    "status": status,
                    "status_info": status_info,
                    "check_successful": status_info.get("check_successful", False)
                }

                print(f"Step {i+1} 状态检查结果: {status}")

            except Exception as e:
                print(f"Step {i+1} 状态检查失败: {e}")
                status_check_result = {
                    "status": "error",
                    "status_info": {"error": str(e)},
                    "check_successful": False
                }
        else:
            status_check_result = {
                "status": "unknown",
                "status_info": {"error": "No agent available"},
                "check_successful": False
            }

        # 保存结果
        step_result = {
            "step": step_data['step'],
            "action": step_data['action_text'],
            "action_json": step_data['action_json'],
            "screen_before": step_data['screen_before'],
            "screen_after": step_data['screen_after'],
            "sub_instruction": inference_result.get("Sub-Instruction", ""),
            "analysis": inference_result.get("Analysis", ""),
            "purpose": inference_result.get("Purpose", ""),
            "high_level_instruction": inference_result.get(
                "High-Level-Instruction", ""
            ),
            "instruction_status": inference_result.get("Status", ""),
            "task_status": status_check_result.get("status", "unknown"),
            "status_check_info": status_check_result.get("status_info", {}),
            "status_check_successful": status_check_result.get("check_successful", False),
            "forward_prompt_token": step_data['forward_prompt_token'],
            "forward_completion_token": step_data['forward_completion_token'],
            "forward_retry_counter": step_data['forward_retry_counter'],
            "reverse_prompt_token": inference_result['reverse_prompt_token'],
            "reverse_completion_token": inference_result['reverse_completion_token'],
            "reverse_retry_counter": inference_result['reverse_retry_counter'],
        }

        return i, step_result

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_step, step_data) for step_data in step_data_list]
        for future in concurrent.futures.as_completed(futures):
            i, res = future.result()
            results[i] = res

    return results
