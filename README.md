# 引力波物理·个人课程资料

20250706

---

## 重要说明

1. 这个库用于存储我学习课程“引力波物理”，完成期末大作业时的各种资料，以代码为主。主要内容是：学习[这个教程](https://gwosc.org/tutorials/)，然后自己上手走一遍流程。同时，我也参考了我们课程的[讲义](https://github.com/yiminghu-SYSU/GW_DA_notes)。
2. 关于数据，我只上传了**我处理后的数据**，缺失的数据和原始数据可以在[这个仓库](https://github.com/gwosc-tutorial/LOSC_Event_tutorial?tab=readme-ov-file)和[GWOSC](https://gwosc.org/events/GW150914/)中找到。
3. 这些代码应该有很多问题，仅供参考。此外，这些代码可以供任意人使用。

---

## 关于整个仓库的说明

1. 大部分代码在`/code`文件夹中，分为预处理、匹配滤波、误警率和时频图几个部分。
2. 用于处理数据的代码被我封装在了`/core/ysy_gw_data_utils.py`中。完整的**笔记**在`/core/note.ipynb`中。
3. 我上传的数据全部在`/data`中，特别说明的是`scores.txt`是误警率评估中处理4096s长的L1原始应变数据得到的结果。

---

## 其它资源

- [用于引力波数据处理的工具包GWpy](https://gwpy.github.io/docs/stable/overview/)；
- [GW170814的辅助通道数据（Witness数据）](https://gwosc.org/auxiliary/GW170814/)；
- [一个时频图+CNN的Kaggle项目](https://github.com/mddunlap924/G2Net_Spectrogram-Classification)；
- [王赫博士的《引力波探测中关于深度学习数据分析的研究》](https://iphysresearch.github.io/PhDthesis_html/)。

