import torch
from torchsummary import summary
from pathlib import Path

from main import get_args
from module import Tokenizer, init_model_by_key

# 需要使用device来指定网络在GPU还是CPU运行
device = torch.device('cuda')
args = get_args()
fdir = Path(args.dir)
tokenizer = Tokenizer.from_pretrained(fdir / 'vocab.pkl')
model = init_model_by_key(args, tokenizer)
model.to(device)
question = '天王盖地虎'
input_ids = torch.tensor(tokenizer.encode(question)).unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(input_ids).squeeze(0)