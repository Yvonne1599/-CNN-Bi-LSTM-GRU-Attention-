from typing import List
from pathlib import Path
import torch


class Tokenizer(object):
    '''一个分词器'''
    def __init__(self):
        # 两个字典，存储单词到索引的映射和索引到单词的映射
        self.token_to_ix = {}
        self.ix_to_token = {}

    # 用于获取词汇表的大小
    @property
    def vocab_size(self):
        return len(self.token_to_ix)

    # 用于获取填充符的索引
    @property
    def pad_id(self):
        return self.token_to_ix['[PAD]']

    # 用于获取未知符的索引
    @property
    def unk_id(self):
        return self.token_to_ix['[UNK]']

    # 用于从一个词汇表文件中构建分词器
    def build(self, vocab_file: str or Path):
        # 判断vocab_file参数是否是一个字符串，如果是，就将其转换为一个Path对象
        if isinstance(vocab_file, str):
            vocab_file = Path(vocab_file)
        # 初始化一个字典，用于存储单词到索引的映射，其中'[PAD]'即填充符对应0，'[UNK]'即未知符对应1
        token_to_ix = {'[PAD]': 0, '[UNK]': 1}
        # 以只读和utf-8编码的方式打开词汇表文件，返回一个文件对象f
        with vocab_file.open('r', encoding='utf-8') as f:
            for token in f.readlines():
                token = token.rstrip("\n")
                if token not in token_to_ix:
                    token_to_ix[token] = len(token_to_ix)
        # 将token_to_ix字典赋值给实例属性token_to_ix，键值对互换实例属性ix_to_token
        self.token_to_ix = token_to_ix
        self.ix_to_token = {v: k for k, v in token_to_ix.items()}

    # 将分词器的信息保存到一个文件中
    def save_pretrained(self, filename: str or Path):
        info_dict = {
            'token_to_ix': self.token_to_ix,
            'ix_to_token': self.ix_to_token,
        }
        if isinstance(filename, str):
            filename = Path(filename)
        filename.parent.mkdir(exist_ok=True, parents=True)
        torch.save(info_dict, filename)

    # 用于从一个文件中加载分词器的信息，返回一个分词器的实例
    @classmethod
    def from_pretrained(cls, filename: str or Path):
        info_dict = torch.load(filename)
        token_to_ix = info_dict['token_to_ix']
        ix_to_token = info_dict['ix_to_token']
        kls = cls()
        kls.token_to_ix = token_to_ix
        kls.ix_to_token = ix_to_token
        return kls

    # 用于将一个文本序列转换为一个数字编码序列
    def encode(self, sent: str):
        tokens = list(sent)
        return self.convert_tokens_to_ids(tokens)

    # 用于将一个数字编码序列转换为一个文本序列
    def decode(self, ids: List[int]):
        tokens = self.convert_ids_to_tokens(ids, True)
        return "".join(tokens)

    # 用于将一个单词转换为一个数字编码
    def convert_token_to_id(self, token: str):
        return self.token_to_ix.get(token, self.token_to_ix['[PAD]'])

    # 用于将一个数字编码转换为一个单词
    def convert_id_to_token(self, id: int):
        return self.ix_to_token[id]

    # 用于将一个单词列表转换为一个数字编码列表
    def convert_tokens_to_ids(self, tokens: List[str]):
        s = []
        for t in tokens:
            i = self.convert_token_to_id(t)
        return [self.convert_token_to_id(t) for t in tokens]

    # 用于将一个数字编码列表转换为一个单词列表
    def convert_ids_to_tokens(self, ids: List[int], ignore_pad: bool = False):
        tokens = []
        for t in ids:
            if ignore_pad and t != self.pad_id:
                tokens.append(self.convert_id_to_token(t))
        return tokens