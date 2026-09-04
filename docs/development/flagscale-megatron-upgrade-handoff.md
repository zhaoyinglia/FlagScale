# FlagScale Megatron-LM-FL v0.18.2 升级交接

## 1. 范围与基线

- FlagScale 工作树：`flagscale/train/megatron/training/`
- upstream 源：`/share/project/zhaoyingli/flagos/Megatron-LM-FL/megatron/training/`
- 目标 upstream commit：`b7acdba0ce95886779e3b05369cd71ea1a89a565`
- 当前 FlagScale 基线：`09295660ba4c325c17d58c9e72cfadbf7538b586`
- 历史功能基线：`megatron/training_fs/`、`megatron/training_v217/`

## 2. 已完成

- 配置结构：引入 v0.18.2 config container/instantiate/yaml/utils。
- 参数层：合并 `arguments.py`、`argument_utils.py`，保留 FlagScale 参数、MLA 兼容和 marker。
- checkpoint：合并新版保存/加载结构，保留平台 RNG、expert-topology 和 FlagScale 扩展。
- utils：迁移为 `utils/common_utils.py`、`utils/log_utils.py`、`utils/__init__.py`，兼容导出已修复。
- 新增/迁移：`activation_logging.py`、`gpu_sniff_test.py`、`vocab_utils.py`、`training/models/*`。
- `training.py`：已完成手工冲突重组并写回；保留 THD/DSv4 FLOPs、新版生命周期及 FlagScale hooks。
- Skill：`skills/flagscale-train-megatron-upgrade/SKILL.md` 已增加 marker、反向差异审计和手工修正证据规则。

## 3. 已验证

在机器 `job-6d37fb70-5816-41bd-b402-05bb2c192aaa-master-0`、环境 `/share/project/zhaoyingli/envs/fs-train`：

- `training` import smoke：通过（`TRAINING_IMPORT_OK`）。
- training 目录全量 `py_compile`：通过。
- `tests/unit_tests/runner/test_path_utils.py`：32 passed。
- `tests/unit_tests/train/utils/test_train_utils.py`：11 passed。
- `flagscale/train/megatron/plugin_flagscale/test_override.py`：5 passed。
- `diffusers`、`pyarrow`、`datasets` 已通过代理安装。

## 4. 未完成/阻塞

- 尚未完成 `FlagScale/tests` 全量回归；CI runner 全量命令启动后被中断。
- 需按 CI 配置执行：

  ```bash
  bash tests/test_utils/runners/run_unit_tests.sh --platform cuda --device a100
  ```

- 该 runner 使用 `torchrun` 和自动检测 GPU 数量，不能用 `WORLD_SIZE=1` 替代并行测试。
- 依赖安装时曾出现 `transformers/tokenizers` 与 `huggingface-hub 1.30.0` 版本约束冲突，应在 CI 镜像中确认兼容版本。
- 需要专项验证 DSv4 FLOPs、异构并行/DualPipeV、checkpoint、PEFT、straggler/perf monitor。

## 5. 注意事项

- 不要用整文件复制或统一选择 ours/theirs 解决 `training.py` 冲突。
- 以 `training_fs`/`training_v217` 和 immutable U0/F0/F1/U1 上下文做反向功能审计。
- marker 只表示 FlagScale 语义边界，不代表功能正确性；需保持成对、最小边界和正确嵌套。
- 格式问题（isort、blank、尾随空格）本次不作为验收条件。

