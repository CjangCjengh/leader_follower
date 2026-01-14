### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Config

复制并修改配置文件：

```bash
cp config.json my_config.json
```

编辑 `my_config.json`，设置API Key、API Base和其他参数

### 3. Run

```bash
python run_avalon_battle.py -c config.json
```

在 watch 模式下，非 `direct` 类型的 Agent 会用青色显示中间思考过程。

## Config说明

```json
{
    "game": {
        "player_nums": 6,           // 玩家数量
        "language": "english",      // 语言 (english/chinese)
        "mode": "watch",            // 模式（watch 模式会显示中间思考过程）
        "game_count": 10,           // 游戏局数
        "start_game_idx": 0,        // 起始游戏索引
        "exp_name": "battle",       // 实验名称
        "camp": "good",             // 阵营 (good/evil/null表示不过滤)
        "output_dir": "logs/avalon/battle",  // 输出目录
        "enable_intent_identification": false  // 是否启用意图识别（识别期望/不期望后置位玩家的发言）
    },
    "default_model": {
        "model_name": "gpt-4o",  // 默认模型
        "api_key": "your-api-key-here",      // API Key
        "api_base": null,                    // API Base
        "temperature": 0.3                   // 温度参数
    },
    "players": [
        {
            "name": "player 1",              // 玩家名称
            "role": null,                    // 角色（null 表示随机分配）
            "agent_type": "direct",          // Agent 类型 (direct/react/recon/lasi)
            "model": {                       // 单独为该玩家配置模型（可选）
                "model_name": "gpt-4o",
                "api_key": "another-api-key",
                "api_base": "https://custom-api.example.com/v1",
                "temperature": 0.5
            }
        },
        // ... 其他玩家
    ],
    "roles": ["Merlin", "Percival", "Loyal Servant", "Loyal Servant", "Morgana", "Assassin"],
    "extractors": {
        "model_name": "gpt-4o",
        "api_key": null,        // null 表示使用 default_model 的配置
        "api_base": null,
        "temperature": 0
    }
}
```

---

## 训练流程

训练流程包含以下步骤：

1. **数据收集**：启用 Intent Identification 运行游戏，收集对话日志
2. **数据转换**：将游戏日志转换为 GRPO 训练格式
3. **启动 Reward Server**：加载 Qwen2.5-72B-Instruct 模型计算 Reward
4. **GRPO 训练**：使用 verl 框架进行训练

### Step 1: 收集数据

首先需要启用 `enable_intent_identification` 开关来生成包含 intent 信息的对话数据。

修改 `config.json`：

```json
{
    "game": {
        ...
        "enable_intent_identification": true  // 启用意图识别
    }
}
```

运行游戏收集数据：

```bash
python run_avalon_battle.py -c config.json
```

每局游戏会生成一个目录，包含 `process.json` 文件。启用 Intent Identification 后，每个讨论事件会包含：
- `desired_responses`: 3 个期望 follower 说的话
- `undesired_responses`: 3 个不期望 follower 说的话

### Step 2: 转换格式

使用转换脚本将游戏日志转换为 verl GRPO 训练格式：

```bash
python scripts/convert_logs_to_grpo_data.py \
    --log_dir logs/avalon/battle \
    --output grpo_training_data.jsonl \
    --include_intent
```

### Step 3: 启动 Reward Server

Reward Server 使用本地 LLM（Qwen2.5-72B-Instruct）作为 Measurer，计算 follower 响应的 log probability

```bash
python scripts/reward_server.py \
    --model_path /path/to/Qwen2.5-72B-Instruct \
    --port 8000 \
    --torch_dtype bfloat16
```

**检查是否启动成功**

```bash
curl http://localhost:8000/health
```

### Step 4: 配置 verl 训练

将 `scripts/rewards.py` 替换到 verl 的 reward function 配置中

```bash
# 配置 Reward Server 地址（如果不是本地 127.0.0.1:8000）
export REWARD_SERVER_HOST=127.0.0.1
export REWARD_SERVER_PORT=8000
```

```bash
# 使用 verl 进行训练
python -m verl.trainer.main_ppo \
    --config your_verl_config.yaml \
    --data_path grpo_training_data.jsonl
```
