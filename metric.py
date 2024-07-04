from typing import List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 创建一个 SmoothingFunction 的实例，用于平滑 BLEU 分数的计算，避免出现零值
sf = SmoothingFunction()

# 计算 BLEU 分数
def calc_bleu(cand: List[int or str], ref: List[int or str]):
    # 调用 sentence_bleu 函数，将参考序列放在一个列表中，作为第一个参数，候选序列作为第二个参数，平滑函数作为第三个参数，返回 BLEU 分数
    return sentence_bleu([ref], cand,weights=(0.5,0.5,0,0), smoothing_function=sf.method1)

# 计算 ROUGE-L 分数
def calc_rouge_l(cand: List[int or str], ref: List[int or str], beta: float = 1.2):
    # 获取候选序列和参考序列的长度
    len_cand = len(cand)
    len_ref = len(ref)
    # 创建一个二维列表，用于存储候选序列和参考序列的最长公共子序列的长度，初始化为零
    lengths = [[0 for j in range(len_ref + 1)] for i in range(len_cand + 1)]
    # 遍历候选序列的每个元素
    for i in range(len_cand):
        # 遍历参考序列的每个元素
        for j in range(len_ref):
            # 如果候选序列和参考序列的当前元素相同，那么最长公共子序列的长度加一，更新二维列表的对应位置
            if cand[i] == ref[j]:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            # 如果不相同，那么最长公共子序列的长度取上方或左方的较大值，更新二维列表的对应位置
            elif lengths[i + 1][j] > lengths[i][j + 1]:
                lengths[i + 1][j + 1] = lengths[i + 1][j]
            else:
                lengths[i + 1][j + 1] = lengths[i][j + 1]
    # 获取二维列表的右下角的值，即为候选序列和参考序列的最长公共子序列的长度
    lcs = lengths[-1][-1]
    # 定义一个很小的正数，用于避免除零错误
    eps = 1e-10
    # 计算 ROUGE-L 的召回率、精确率
    r = lcs * 1.0 / (eps + len_ref)
    p = lcs * 1.0 / (eps + len_cand)
    # 计算 ROUGE-L 的 F 值，即召回率和精确率的加权调和平均，beta 为权重系数，越大表示越偏重召回率
    f = ((1 + beta**2) * r * p) / (eps + r + beta ** 2 * p)
    return f
