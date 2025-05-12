# GLM-Edge-V-2B--Monitoring_Species
Built with GLM-Edge

<p align="center">
👋 联系作者在工作邮箱：Raymond.dengruihan@edu.yungu.org
</p>
<p align="center">
或个人邮箱：551310554@qq.com
</p>
<p align="center">
或个人电话：18368725059
</p>
<p align="center">
工作邮箱不一定联系得上，如果没有及时回复请考虑另外的联系方式
</p>

## 引言
这是一个基于由智谱AI开源的端侧多模态大模型GLM-Edge-V-2b的福寿螺卵自动统计系统
本仓库的所有者为该项目的主要开发人员
该项目由2024-2025届CTB（China Thinks Big）比赛，”影视旋风“小队的创新项目发展而来，我们小队奋斗半年，最终产出了这样一个项目，我将会把我们在比赛期间内所完成的部分保存在本仓库的”Original“文件夹，保留一份我们的青春回忆以及奋斗历史。在这里感谢与我并肩作战的老师及同学们：Jie, Felcia, Alice, Carl, Robin。

## 项目更新
- 2025年5月：上传了```app.py```和```information.py```，替换了了其中的文件路径，使用中文提示
- 2025年4月：上传```original```文件，该源文件直接复制自CTB竞赛时期的电脑文件夹

## 测试环境
|           Model            |   Type    | Seq Length* |                                                                                                                                                              Download                                                                                                                                                              |
|:--------------------------:|:---------:|:-----------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|       GLM-4-9B-0414        |   Chat    | 32K -> 128K |                           [🤗 Huggingface](https://huggingface.co/THUDM/GLM-4-9B-0414)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-4-9B-0414)<br> [🧩 Modelers](https://modelers.cn/models/zhipuai/GLM-4-9B-0414)<br> [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/GLM-4-9B-0414)                           |
|       GLM-Z1-9B-0414       | Reasoning | 32K -> 128K |                        [🤗 Huggingface](https://huggingface.co/THUDM/GLM-4-Z1-9B-0414)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-Z1-9B-0414)<br> [🧩 Modelers](https://modelers.cn/models/zhipuai/GLM-Z1-9B-0414)<br> [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/GLM-Z1-9B-0414)                        |
|    GLM-4-32B-Base-0414     |   Base    | 32K -> 128K |               [🤗 Huggingface](https://huggingface.co/THUDM/GLM-4-32B-Base-0414)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-4-32B-Base-0414)<br> [🧩 Modelers](https://modelers.cn/models/zhipuai/GLM-4-32B-Base-0414)<br> [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/GLM-4-32B-Base-0414)               |
|       GLM-4-32B-0414       |   Chat    | 32K -> 128K |                      [🤗 Huggingface](https://huggingface.co/THUDM/GLM-4-32B-0414)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-4-32B-0414)<br> [🧩 Modelers](https://modelers.cn/models/zhipuai/GLM-4-32B-0414)<br> [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/GLM-4-32B-Base-0414)                       |
|      GLM-Z1-32B-0414       | Reasoning | 32K -> 128K |                       [🤗 Huggingface](https://huggingface.co/THUDM/GLM-Z1-32B-0414)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-Z1-32B-0414)<br> [🧩 Modelers](https://modelers.cn/models/zhipuai/GLM-Z1-32B-0414)<br> [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/GLM-Z1-32B-0414)                       |
| GLM-Z1-Rumination-32B-0414 | Reasoning |    128K     | [🤗 Huggingface](https://huggingface.co/THUDM/GLM-Z1-Rumination-32B-0414)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-Z1-Rumination-32B-0414)<br> [🧩 Modelers](https://modelers.cn/models/zhipuai/GLM-Z1-Rumination-32B-0414)<br> [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/GLM-Z1-Rumination-32B-0414) |

GLM-4-9B-0414 由于其较小的模型容量，我们未对其智能体能力进行类似 GLM-4-32B-0414 的强化，主要针对翻译等需要大批量调用的场景进行优化。

\* 模型原生采用 32K 上下文进行训练，对于输入 + 输出长度可能超过 32K 的请求，我们建议激活 YaRN 来获得较好的外推性能，详情见[部署章节](#%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%8F%90%E7%A4%BA%E8%AF%8D%E5%AE%9E%E7%8E%B0)。

以下为 2024 年 6 月 5 日发布的 GLM-4 系列模型，其详细内容可以在[这里](README_zh_240605.md)查看。

|             Model             |   Type    | Seq Length* |                                                                                                      Download                                                                                                       |
|:-----------------------------:|:---------:|:----------:|
## Quick Start
