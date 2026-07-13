# AiParking 模型训练与监督交接文档

> 适用项目：`D:\Aiparking\Aiparking For YOLO`  
> 当前数据主目录：`D:\Aiparking\IImages`  
> 当前主线：YOLOv8n-seg、512 输入、Parking 单类别、四边形训练标注  
> 最后核对日期：2026-07-13

本文档用于后续人员或自动化代理接管 AiParking 的数据清理、训练、监督、验证、部署文件导出和 GitHub 记录工作。任何新一轮训练都必须先阅读“红线要求”和“训练前检查清单”。

---

## 1. 当前可交付基线

当前场景专用模型由 `D:\Aiparking\IImages` 训练得到，只包含 `Parking` 类，不包含 `barrier`。

| 项目 | 当前值 |
|---|---|
| 模型 | YOLOv8n-seg |
| 输入尺寸 | 512×512 |
| 类别 | `0: Parking` |
| 训练标签 | 四边形 polygon，4 个顶点 |
| 训练集 | 5,604 张，4,448 个实例，1,186 张负样本 |
| 验证集 | 1,867 张，1,814 个实例，74 张负样本 |
| 实际训练 | 344/500 epochs，EarlyStopping |
| 最佳轮次 | epoch 264 |
| 训练时长 | 6.981 小时 |
| Box mAP50 / mAP50-95 | 0.987 / 0.893 |
| Mask mAP50 / mAP50-95 | 0.986 / 0.836 |
| 参数量 / 计算量 | 3,258,259 / 11.3 GFLOPs |

稳定部署文件：

- `models\best_yolov8n_iimages_quad_v1.pt`
- `models\best_yolov8n_iimages_quad_v1_512.onnx`

ONNX 结构：

- 输入：`images=(1,3,512,512)`，float32，BCHW
- 输出：`output0=(1,37,5376)`
- 输出：`output1=(1,32,128,128)`

注意：本模型用四边形标注训练，但 YOLOv8n-seg 原始输出仍是检测结果和分割掩码，不保证直接输出固定四个角点。这是已明确接受的项目决策，不得擅自改为 OBB、Pose 或其他模型。

---

## 2. 目录职责

```text
D:\Aiparking\Aiparking For YOLO
├── train.py                              # YOLOv8-seg 训练和训练后 ONNX 导出
├── build_iimages_yolov8_dataset.py       # IImages 专用递归数据集构建器
├── monitor_iimages_training.ps1          # 每小时训练/GPU 监督脚本
├── models\                               # 稳定、可交付、带版本名的模型
├── runs\                                 # Ultralytics 原始训练输出，不上传 GitHub
├── log\                                  # 中文 Markdown 日志；原始 .log 不上传 GitHub
├── tests\                                # 自动化测试
├── CHANGELOG.md                           # 版本更新记录
└── .gitignore                             # 大型素材、runs、原始日志和临时模型规则

D:\Aiparking\IImages
├── 各场景\images\                        # 图片和同名 X-AnyLabeling JSON
├── 各场景\raw_images\                    # 仅当图片有同名 JSON 时可训练
└── dataset_yolov8n_quad_v1\              # 生成的 YOLO 数据集，不上传 GitHub

D:\Aiparking\Aiparking For YOLObackup    # 默认训练前备份目录
```

`prelabel_quad\annotations_quad` 没有同目录图片，是派生/重复标注目录，不得与主 `images` 标注同时加入训练。

---

## 3. 红线要求

以下规则属于“没有明确书面豁免就绝对不能做”的红线。

### 3.1 原始素材红线

1. **禁止删除、移动、覆盖原始图片和 X-AnyLabeling JSON。** 数据集构建只能输出到新的 `dataset_yolov8n*` 目录。
2. **禁止把没有同名 JSON 的图片当作负样本。** 无 JSON 表示未知状态，不等于“确认没有车位”。
3. **禁止把有 shapes、但因低置信度或几何错误被全部过滤的图片写成空标签。** 这种图片必须排除，不能伪造成背景图。
4. **禁止同时使用主 JSON 和 `prelabel_quad\annotations_quad`。** 否则同一素材会重复计数并造成数据泄漏。
5. **禁止静默接受未知标签。** `Parking` 和 `parking` 可统一为 `Parking`；其他标签必须统计、确认后才能处理。
6. **禁止把自动预测的空 JSON 默认视为人工确认负样本。** 必须抽样查看；只有确认画面无完整目标后才允许纳入。
7. **禁止把低于 0.4 的自动标注用于训练。** 无 `score` 字段的人工标注保留；不能给人工标注伪造置信度。

### 3.2 标注与类别红线

1. 当前 `IImages` 模型只训练 `Parking`。没有 `barrier` 样本时，禁止保留一个空的 barrier 类别。
2. Parking 标签必须最终转换为四边形 4 点 polygon。不能退回矩形框训练。
3. 5 点及以上 polygon 必须先检查；构建器只允许用最小局部几何误差规范化为 4 点，不得随机删点。
4. 所有坐标必须裁剪到图像边界并归一化到 `[0,1]`；发现 NaN、Inf、少于 4 点时必须排除并计数。
5. 修改类别顺序、类别名称、输入尺寸或模型结构，必须同步修改数据 YAML、部署代码、ONNX 说明和 CHANGELOG。

### 3.3 训练红线

1. **默认必须先备份再训练。** 备份位置固定为 `D:\Aiparking\Aiparking For YOLObackup`。只有用户针对某一轮明确书面豁免时才能跳过，并必须写入该轮日志。
2. **默认不得跳过去重或数据审计。** 2026-07-13 的“不去重、不备份”是一次性明确豁免，不构成永久流程。
3. **禁止误用 CPU 长时训练。** 启动前必须看到 `torch.cuda.is_available() == True` 和 RTX 4060；否则立即停止。
4. **禁止只看任务管理器的 3D 曲线判断训练状态。** CUDA 训练应使用 `nvidia-smi` 或任务管理器的 CUDA/Compute 图表。
5. **禁止把连续帧随机分散到训练集和验证集。** 必须按场景/拍摄序列分组划分，避免验证指标虚高。
6. **禁止覆盖稳定模型。** 每轮输出必须使用新的 run name 和版本化模型名；确认验收后再复制到 `models`。
7. **禁止仅凭 mAP50 宣布模型可部署。** 必须同时检查 Recall、mAP50-95、真实图片预测和板端延迟。
8. 出现 loss=NaN/Inf、CUDA 不可用、数据实例为 0、持续 OOM、图片损坏或 results.csv 长时间不更新时，必须先停止并查明原因，禁止盲目重启。

### 3.4 Git 与部署红线

1. 禁止把原始图片、生成数据集、`runs`、epoch 中间权重和原始 stdout/stderr 日志提交到 GitHub。
2. 只允许把最终稳定 `.pt/.onnx` 加入 `models`，并在 `.gitignore` 中为具体文件名增加白名单。
3. 禁止使用 `git add -f` 绕过忽略规则提交大型素材。
4. ONNX 的输入尺寸、opset、类别数或输出结构变化后，禁止沿用旧文件名和旧板端配置。
5. SSH 端口 22 被网络关闭时，可使用 HTTPS 推送；禁止通过重写历史或强推来“解决”普通连接问题。

---

## 4. 默认训练要求

没有新的书面决策时，IImages 场景模型采用以下固定参数：

| 参数 | 要求 |
|---|---|
| Python | `C:\Users\ZhanTu Shen\.conda\envs\yolov11\python.exe` |
| 框架 | Ultralytics 8.4.21 |
| PyTorch | 2.5.1+cu121 |
| GPU | NVIDIA GeForce RTX 4060 Laptop，CUDA:0 |
| 初始模型 | `yolov8n-seg.pt`，不是旧项目 best.pt |
| 任务 | segment |
| 类别 | `Parking` 单类别 |
| 输入尺寸 | 512 |
| batch | 32；OOM 时先降到 24/16，不得切 CPU |
| epochs | 500 上限 |
| patience | 80 |
| workers | 4 |
| save_period | 25 |
| min_conf | 0.4，仅针对有 score 的自动标注 |
| plots | False，规避当前 polars CPU feature 异常 |
| ONNX | opset 12、simplify=True、FP32 |

目标训练时长约 6–8 小时。实际结束条件由 EarlyStopping 决定，不要求机械跑满 500 轮。

---

## 5. 标准训练流程

### 5.1 第一步：训练前备份

除非本轮有明确豁免，否则先备份。使用镜像复制前必须人工确认源和目标字符串完全正确。

```powershell
$source = 'D:\Aiparking\Aiparking For YOLO'
$backup = 'D:\Aiparking\Aiparking For YOLObackup'

if ($source -eq $backup -or $backup -ne 'D:\Aiparking\Aiparking For YOLObackup') {
    throw '备份路径不安全，停止执行。'
}

robocopy $source $backup /MIR /XD .git __pycache__ /XF *.pyc
if ($LASTEXITCODE -gt 7) {
    throw "备份失败，Robocopy exit=$LASTEXITCODE"
}
```

`/MIR` 会覆盖旧备份内容，只能对固定备份目录使用。禁止把目标变量指向项目目录、数据源目录或磁盘根目录。

### 5.2 第二步：只读数据审计

至少确认以下内容并写入当轮 Markdown 日志：

- 图片总数、JSON 总数、同目录同名配对数
- 无 JSON 图片数、无图片 JSON 数、JSON 解析失败数
- 空 JSON 数、正样本数、实例数
- 标签名称和大小写分布
- shape_type 和顶点数量分布
- 自动标注 score 的最小值、低于 0.4 的数量
- 图片尺寸和越界坐标数量
- 是否存在 `prelabel_quad`、manifest、pipeline metadata 等非训练 JSON

判断原则：JSON 数量大于图片数量不代表标注更多，常见原因是 metadata 和派生标注目录。

### 5.3 第三步：构建 IImages 数据集

```powershell
$python = 'C:\Users\ZhanTu Shen\.conda\envs\yolov11\python.exe'
$project = 'D:\Aiparking\Aiparking For YOLO'

& $python "$project\build_iimages_yolov8_dataset.py" `
  --data-root 'D:\Aiparking\IImages' `
  --output 'D:\Aiparking\IImages\dataset_yolov8n_quad_v2' `
  --min-conf 0.4 `
  --val-ratio 0.15 `
  --seed 20260713
```

新一轮必须使用新的输出目录版本，如 `quad_v2`、`quad_v3`。构建器会清空同名 `dataset_yolov8n*` 输出，因此不能把输出目录指向原始素材。

构建后必须核对：

```powershell
Get-Content 'D:\Aiparking\IImages\dataset_yolov8n_quad_v2\build_summary.json'
```

必须满足：

- `train_images > 0`
- `val_images > 0`
- 图片数量与 TXT 标签数量完全一致
- `instances > 0`
- unknown_label、bad_json、bad_image、non_quadrilateral_shapes 均已解释
- 验证集按完整场景组隔离，不是逐帧随机泄漏

### 5.4 第四步：确认 CUDA

```powershell
& $python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

若输出不是 `True` 和 RTX 4060，禁止启动 6–8 小时训练。

### 5.5 第五步：启动训练

前台启动命令最简单、最不容易产生路径解析问题：

```powershell
& $python "$project\train.py" `
  --data 'D:\Aiparking\IImages\dataset_yolov8n_quad_v2\parking_yolov8.yaml' `
  --model "$project\yolov8n-seg.pt" `
  --epochs 500 `
  --imgsz 512 `
  --batch 32 `
  --patience 80 `
  --workers 4 `
  --project "$project\runs" `
  --name 'parking_yolov8n_iimages_quad_v2' `
  --save-period 25
```

后台启动时，`Start-Process -ArgumentList` 中所有带空格路径必须再次加双引号：

```powershell
$stdout = "$project\log\YYYY-MM-DD_iimages_train_stdout.log"
$stderr = "$project\log\YYYY-MM-DD_iimages_train_stderr.log"
$trainArgs = '"{0}" --data "{1}" --model "{2}" --epochs 500 --imgsz 512 --batch 32 --patience 80 --workers 4 --project "{3}" --name "{4}" --save-period 25' -f `
  "$project\train.py", `
  'D:\Aiparking\IImages\dataset_yolov8n_quad_v2\parking_yolov8.yaml', `
  "$project\yolov8n-seg.pt", `
  "$project\runs", `
  'parking_yolov8n_iimages_quad_v2'

$training = Start-Process `
  -FilePath $python `
  -ArgumentList $trainArgs `
  -WorkingDirectory $project `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -WindowStyle Hidden `
  -PassThru

$training.Id
```

若 stderr 出现 `can't open file 'D:\Aiparking\Aiparking'`，说明带空格路径没有正确引用，训练实际上没有启动。

---

## 6. 每小时监督脚本

脚本：`monitor_iimages_training.ps1`

### 6.1 脚本做什么

每隔 `IntervalSeconds` 秒执行一次：

1. 检查指定训练 PID 是否仍存在。
2. 读取 run 目录下 `results.csv` 最后一行。
3. 记录 epoch、Box/Mask mAP50、Box/Mask mAP50-95。
4. 调用 `nvidia-smi` 记录 GPU 利用率、显存和温度。
5. 训练 PID 消失后写入结束状态并退出。

它只负责记录，不会自动重启训练、修改参数或向聊天窗口主动报警。

### 6.2 启动监督脚本

因为项目路径有空格，推荐使用 EncodedCommand 启动后台监督：

```powershell
$monitorScript = "$project\monitor_iimages_training.ps1"
$runDirectory = "$project\runs\parking_yolov8n_iimages_quad_v2"
$monitorLog = "$project\log\YYYY-MM-DD_iimages_monitor.log"
$monitorOut = "$project\log\YYYY-MM-DD_iimages_monitor_stdout.log"
$monitorErr = "$project\log\YYYY-MM-DD_iimages_monitor_stderr.log"

$monitorCommand = "& '$monitorScript' -TargetPid $($training.Id) -RunDirectory '$runDirectory' -MonitorLog '$monitorLog' -IntervalSeconds 3600"
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($monitorCommand))

$monitor = Start-Process `
  -FilePath 'powershell.exe' `
  -ArgumentList "-NoProfile -ExecutionPolicy Bypass -EncodedCommand $encoded" `
  -WorkingDirectory $project `
  -RedirectStandardOutput $monitorOut `
  -RedirectStandardError $monitorErr `
  -WindowStyle Hidden `
  -PassThru

$monitor.Id
```

启动后 5–10 秒检查：

```powershell
Get-Content $monitorLog -Tail 5
Get-Content $monitorErr -Tail 20
```

正常首条记录类似：

```text
[2026-07-13 03:40:37] epoch=1; box_mAP50=...; mask_mAP50=...; gpu=81, 5461, 8188, 66; training is running
```

### 6.3 监督脚本限制

- 监督脚本绑定主训练 PID。训练重启或真正 resume 后必须重新启动监督脚本。
- 训练结束状态最多延迟一个监督周期才会写入；训练结束后应人工立即核验。
- GPU 利用率是瞬时值，会在数据加载、验证和 batch 间隙下降，单次低值不代表没有训练。
- `results.csv not available` 在第一个 epoch 完成前属于正常状态。
- 原始 `.log` 被 Git 忽略；关键结论必须写入中文 Markdown 日志。

---

## 7. 训练中判断规则

每小时检查至少报告：当前 epoch、过去一小时轮数、Box/Mask mAP50-95、GPU 显存、温度和预计剩余时间。

### 正常现象

- 第 1–5 轮指标剧烈波动。
- GPU 利用率在 10%–90% 间瞬时变化。
- 验证阶段 GPU 利用率下降。
- ONNX 导出日志显示 CPU；导出使用 CPU 不代表训练使用核显。
- mAP50 较早达到高值，而 mAP50-95 继续缓慢提升。

### 必须暂停检查的情况

- `torch.cuda.is_available()` 为 False，或日志显示 `device=cpu`。
- CUDA OOM 连续发生。
- 任意 loss 变为 NaN/Inf。
- 日志显示 0 instances、No labels found 或大量 corrupt images。
- results.csv 两小时没有新增，训练 PID CPU/GPU 同时长期空闲。
- GPU 温度持续超过 85°C，应暂停并检查散热、电源模式和风道；最终限制以硬件厂商规范为准。
- mAP 和 Recall 长期接近 0；应检查类别 ID、YAML 和标签格式，而不是增加 epochs。

不要因为某一轮指标下降就停止。EarlyStopping 已负责判断长期无提升。

---

## 8. 训练结束验收

训练完成不能只看进程消失，必须逐项验证：

1. stdout 中出现 `Training finished` 或 EarlyStopping 完整结束信息。
2. `runs\<run-name>\weights\best.pt` 和 `last.pt` 存在且大小合理。
3. stdout 显示使用 `best.pt` 完成最终验证。
4. 记录实际训练轮数、最佳轮次、耗时、Precision、Recall、mAP50、mAP50-95。
5. `best.onnx` 成功导出。
6. ONNX 输入和输出结构通过检查。
7. 用真实未参与训练的图片进行一次预测抽查。
8. 复制稳定模型到 `models`，使用版本化名称，不能覆盖旧稳定版本。
9. 更新 `log\YYYY-MM-DD_*.md` 和 `CHANGELOG.md`。
10. 运行测试和 Git 检查，再提交 GitHub。

ONNX 检查命令：

```powershell
& $python -c "import onnx; p=r'D:\Aiparking\Aiparking For YOLO\models\MODEL.onnx'; m=onnx.load(p); onnx.checker.check_model(m); print([(i.name,[d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]); print([(o.name,[d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])"
```

测试命令：

```powershell
& $python -m unittest discover -s "$project\tests"
git -C $project diff --check
git -C $project status --short
```

---

## 9. 中断与恢复

### 9.1 区分“继续微调”和“真正恢复”

把 `last.pt` 作为 `--model` 重新运行 `train.py` 只是从该权重重新微调，不会恢复原优化器、epoch 和 EarlyStopping 状态。

真正 resume 应使用 Ultralytics 的 `resume=True`：

```powershell
& $python -c "from ultralytics import YOLO; YOLO(r'D:\Aiparking\Aiparking For YOLO\runs\RUN_NAME\weights\last.pt').train(resume=True)"
```

resume 后训练 PID 会改变，必须重新启动监督脚本。

### 9.2 不允许恢复的情况

- 数据 YAML、类别顺序、输入尺寸或模型结构已经改变。
- 原 run 的 args.yaml 与当前计划不一致。
- last.pt 损坏或来源不明。

这些情况应新建 run，从明确的官方预训练模型或已审核稳定模型开始。

---

## 10. 已知易错点

| 易错点 | 表现 | 正确处理 |
|---|---|---|
| Python 环境错误 | `Invalid CUDA device=0`、torch 为 CPU 版 | 使用固定 `yolov11\python.exe`，先检查 CUDA |
| 路径含空格 | `can't open file 'D:\Aiparking\Aiparking'` | Start-Process 的参数内部再次加引号，或用 EncodedCommand |
| 把 JSON 总数当标注数 | JSON 比图片多 | 排除 manifest、pipeline metadata、prelabel_quad 派生目录 |
| 把无 JSON 图片当负样本 | 模型学到错误背景 | 只允许“图片+同名空 JSON”作为候选负样本，并抽样确认 |
| 低置信度过滤后写空标签 | 有目标图片被教成背景 | 有 shapes 但全部被过滤时，整张图片排除 |
| 标签大小写拆成两类 | `Parking` 和 `parking` 类别错位 | 构建时统一为 `Parking` |
| 5 点 polygon 直接训练 | 不满足四边形要求 | 去重相邻点，并按最小局部几何误差降为 4 点 |
| 训练/验证连续帧泄漏 | mAP 虚高、换场景失效 | 按 Ground/拍摄场景分组划分 |
| 只看任务管理器 3D | 误判 GPU 没工作 | 使用 nvidia-smi 或 CUDA/Compute 图表 |
| polars `sse3` 异常 | 训练完成后绘图/读 CSV 报错 | train.py 已安装 CSV fallback 且 plots=False；先确认权重和验证是否完成 |
| ONNX 显示 CPU | 误判训练用了核显 | ONNX 导出可用 CPU；训练日志必须显示 CUDA:0 |
| 监督脚本未产生日志 | PowerShell 解析错误或路径引用错误 | 先以前台假 PID测试语法，再后台启动，检查 monitor stderr |
| 覆盖 best.onnx | 多尺寸导出互相覆盖 | 使用带尺寸/版本名的模型文件 |
| GitHub SSH 失败 | port 22 connection closed | 改用 HTTPS push，不强推、不改历史 |
| `.git` 变得很大 | 原图、runs 或中间权重进历史 | 遵守 .gitignore，只提交稳定模型和 Markdown 日志 |

---

## 11. GitHub 收尾要求

每轮稳定模型确认后：

1. 在 `.gitignore` 为最终版本化 `.pt/.onnx` 添加精确白名单。
2. 不提交 `runs`、数据集、原始 `.log` 和 epoch 中间权重。
3. 中文日志和 CHANGELOG 必须包含数据来源、参数、耗时、最佳 epoch、最终指标和部署文件名。
4. 提交前运行 19 个测试、ONNX checker 和 `git diff --check`。
5. 推送后使用 `git ls-remote` 确认远端 master 指向本地最新提交。

推荐命令：

```powershell
git -C $project status --short
git -C $project add -- <明确文件列表>
git -C $project commit -m "Add <版本> AiParking model"
git -C $project push https://github.com/shenzhantu/Aiparking-For-YOLO.git master
git ls-remote https://github.com/shenzhantu/Aiparking-For-YOLO.git refs/heads/master
```

禁止在不理解远端差异时使用 `--force` 或 `--force-with-lease`。

---

## 12. 每轮交接清单

完成一轮训练后，交接人必须能够回答：

- 本轮训练用了哪些具体目录？
- 图片/JSON 有效配对多少？正样本、负样本、实例多少？
- 标签和类别是否发生变化？
- 是否执行备份和去重？若没有，书面豁免在哪里？
- train/val 如何隔离场景？验证集包含哪些组？
- 使用哪个 Python、PyTorch、CUDA、GPU？
- 模型、imgsz、batch、epochs、patience 是多少？
- 实际跑了多少轮、多久、最佳轮次是多少？
- Box/Mask 的 Precision、Recall、mAP50、mAP50-95 是多少？
- PT/ONNX 在哪里？ONNX 输入输出是什么？
- 真实图片抽查结果如何？
- 中文日志、CHANGELOG、测试和 GitHub 是否完成？

任何一项无法回答，都不能把该模型标记为“稳定可交付”。
