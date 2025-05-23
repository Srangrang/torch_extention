# coding=utf-8
# Copyright 2022 HuggingFace Inc. team and BigScience workshop.
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

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

# Copyright (c) 2021 EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# 解析命令行参数
parser = argparse.ArgumentParser(description='与模型进行对话')
parser.add_argument('--question', type=str, required=True, help='向模型提问的问题')
args = parser.parse_args()

# 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained('TeleChat-12B', trust_remote_code=True, from_tf=True)
model = AutoModelForCausalLM.from_pretrained('TeleChat-12B', trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
generate_config = GenerationConfig.from_pretrained('TeleChat-12B')

# 获取问题
question = args.question

# 生成答案
answer, history = model.chat(tokenizer=tokenizer, question=question, history=[], generation_config=generate_config, stream=False)
print("TeleChat:" + answer)