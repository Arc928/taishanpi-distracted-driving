# Grounding DINO 开放词预标注复核集

本目录是独立复核副本，来源为 multiview_statefarm_3mdad_v1，原目录未改动。
State Farm 1534 张原标签及来源原样继承；3MDAD train 1540 / val 220 / test 440 共 2200 张已完成开放词预标注。
使用 X-AnyLabeling 支持的 Grounding DINO SwinT 同系列官方非量化权重，通过 PyTorch 批处理并输出 X-AnyLabeling JSON。
所有图片仍待人工复核，不存在自动接受的检测标签，也没有可直接训练的 data.yaml。

## 打开与复核

在 X-AnyLabeling 依次打开 review/train/train_01 等批次，之后 review/val/val_01、review/test/test_01 和 test_02。
固定类别为 0 texting、1 calling。texting 框持机手和手机，不含扶方向盘的手；calling 框持机手、手机及附近脸部。
必须检查司机/乘客、漏检、多框、框范围和真实动作。模型可能把方向盘、手或其他物体误当手机。
正样本类别来自原动作弱标签，背景候选的类别只是几何猜测，均须人工修正；框是检测部件的启发式合并。
原 .mat 头手标签未用于生成这些框。

保存 JSON 后，在本目录 review_status.csv 把每张图 decision 从 pending 改为 positive/background/exclude，并为 exclude 填 note。
source AC5–AC8 无框不能改 background，应补框或 exclude。背景候选也必须逐张看过；没有框不代表确认背景。
val/test 可以接受辅助预填，但必须全部人工复核，尤其检查漏检和错误预填，不能把这次预测当真值。
使用相同固定参数处理所有 split；不能根据 test 的模型成绩反复调提示词、阈值或标注标准。

## 后续导出

沿用项目 export_multiview.py，--dataset 指向本目录，--output 指向新的导出目录，并记录真实复核完成确认。
导出器会校验冻结清单，拒绝尚未完整审核的 val，只有明确完成审核的图片会导出。
test 完整审核前不会写入训练配置。旧 State Farm p016/p021/p026 继续单独保留为原图像级 eval。

grounding_dino_predictions.jsonl 保留原始部件检测，model_provenance.json 保留权重哈希、提示词及阈值。
review_initial_hashes.json 是本轮新预填的哈希基线，和原自训练模型的预填不同。
许可沿用原 3MDAD CC BY-NC-ND 4.0，教育/研究、非商业用途。
