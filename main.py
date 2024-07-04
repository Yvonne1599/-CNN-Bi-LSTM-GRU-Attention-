import argparse
import logging
from pathlib import Path
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from module import Tokenizer, init_model_by_key
from module.metric import calc_bleu, calc_rouge_l

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 获取参数
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=768, type=int)
    parser.add_argument("--max_seq_len", default=32, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("-m", "--model", default='bilstmconvattres', type=str)
    parser.add_argument("--max_grad_norm", default=3.0, type=float)
    parser.add_argument("--dir", default='dataset', type=str)
    parser.add_argument("--output", default='output', type=str)
    parser.add_argument("--logdir", default='runs', type=str)
    parser.add_argument("--embed_dim", default=128, type=int)
    parser.add_argument("--n_layer", default=1, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--ff_dim", default=512, type=int)
    parser.add_argument("--n_head", default=8, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)

    parser.add_argument("--re", default=False, type=bool)
    parser.add_argument("--re_path", default='', type=str)
    parser.add_argument("--test_epoch", default=1, type=int)
    parser.add_argument("--save_epoch", default=10, type=int)

    parser.add_argument("--embed_drop", default=0.2, type=float)
    parser.add_argument("--hidden_drop", default=0.1, type=float)
    return parser.parse_args()

# 用于自动评估模型的性能
def auto_evaluate(model, testloader, tokenizer):
    # 创建一个空列表bleus，存储每个样本的BLEU分数、Rouge-L分数
    bleus = []
    rls = []
    device = next(model.parameters()).device
    # 将模型设置为评估模式，不进行梯度更新
    model.eval()
    for step, batch in enumerate(testloader):
        input_ids, masks, lens = tuple(t.to(device) for t in batch[:-1])
        target_ids = batch[-1]
        # 不计算梯度，节省内存
        with torch.no_grad():
            logits = model(input_ids, masks)
            _, preds = torch.max(logits, dim=-1) 
        # 遍历每个预测序列和目标序列
        for seq, tag in zip(preds.tolist(), target_ids.tolist()):
            # 去掉填充符
            seq = list(filter(lambda x: x != tokenizer.pad_id, seq))
            tag = list(filter(lambda x: x != tokenizer.pad_id, tag))
            # 计算预测序列和目标序列的BLEU分数、Rouge-L分数
            bleu = calc_bleu(seq, tag)
            rl = calc_rouge_l(seq, tag)
            bleus.append(bleu)
            rls.append(rl)
    # 返回所有样本的BLEU分数和Rouge-L分数的平均值
    return sum(bleus) / len(bleus), sum(rls) / len(rls)

# 选择一些样本，用模型进行预测，并输出预测结果
def predict_demos(model, tokenizer:Tokenizer):
    demos = [
        "马齿草焉无马齿", "天古天今，地中地外，古今中外存天地", 
        "笑取琴书温旧梦", "日里千人拱手划船，齐歌狂吼川江号子",
        "我有诗情堪纵酒", "我以真诚溶冷血",
        "三世业岐黄，妙手回春人共赞"
    ]
    sents = [torch.tensor(tokenizer.encode(sent)).unsqueeze(0) for sent in demos]
    model.eval()
    device = next(model.parameters()).device
    for i, sent in enumerate(sents):
        sent = sent.to(device)
        with torch.no_grad():
            logits = model(sent).squeeze(0)
        pred = logits.argmax(dim=-1).tolist()
        pred = tokenizer.decode(pred)
        logger.info(f"上联：{demos[i]}。 预测的下联：{pred}")

# 保存模型、参数、分词器
def save_model(filename, model, args, tokenizer):
    info_dict = {
        'model': model.state_dict(),
        'args': args,
        'tokenzier': tokenizer
    }
    torch.save(info_dict, filename)

# 用于运行模型的训练和测试
def run():
    args = get_args()
    fdir = Path(args.dir)
    # 创建一个tensorboard的写入器，用于记录训练过程中的信息
    tb = SummaryWriter(args.logdir)
    # 获取设备类型，是CPU还是GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # 获取输出文件的路径
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    # 用日志记录器输出
    logger.info(args)
    logger.info(f"loading vocab...")
    # 从文件中加载分词器
    tokenizer = Tokenizer.from_pretrained(fdir / 'vocab.pkl')
    logger.info(f"loading dataset...")
    # 从文件中加载训练数据集和测试数据集
    train_dataset = torch.load(fdir / 'train.pkl')
    test_dataset = torch.load(fdir / 'test.pkl')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    if args.re:
        logger.info(f"initializing trained model...")
        model_info = torch.load(args.re_path)
        model = init_model_by_key(model_info['args'], tokenizer)
        model.load_state_dict(model_info['model'])
        model.to(device)
    else:
        logger.info(f"initializing model...")
        # 根据命令行参数和分词器初始化模型
        model = init_model_by_key(args, tokenizer)
        # 将模型转移到设备上
        model.to(device)
    # 定义损失函数，使用交叉熵损失，忽略填充符的影响
    loss_function = nn.CrossEntropyLoss(reduction='mean',ignore_index=tokenizer.pad_id)
    # 定义优化器，使用Adam优化器，指定学习率
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # 如果有多个GPU，就使用数据并行的方式训练模型
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # 定义学习率调整器，根据损失值的变化来调整学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    logger.info(f"num gpu: {torch.cuda.device_count()}")
    # 初始化全局步数为0
    global_step = 0
    for epoch in range(args.epochs):
        logger.info(f"***** Epoch {epoch} *****")
        model.train()
        t1 = time.time()
        # 初始化累积的损失值为0.0
        accu_loss = 0.0
        for step, batch in enumerate(train_loader):
            # 清空优化器的梯度
            optimizer.zero_grad()
            # 将数据转移到设备上
            batch = tuple(t.to(device) for t in batch)
            input_ids, masks, lens, target_ids = batch
            logits = model(input_ids, masks)
            # 计算损失值，使用交叉熵损失，忽略填充符的影响
            loss = loss_function(logits.view(-1, tokenizer.vocab_size), target_ids.view(-1))
            # 如果有多个GPU，就取损失值的平均值
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            # 将损失值累加到累积的损失值上
            accu_loss += loss.item()
            LOSS.append(loss.item())
            # 反向传播，计算梯度
            loss.backward()
            # 对模型的参数进行梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # 更新模型的参数
            optimizer.step()
            # 如果当前的步数是100的倍数，就记录损失值到tensorboard
            if step % 100 == 0:
                tb.add_scalar('loss', loss.item(), global_step)
                logger.info(
                    f"[epoch]: {epoch}, [batch]: {step}, [loss]: {loss.item()}")
            global_step += 1
        # 根据累积的损失值调整学习率
        scheduler.step(accu_loss)
        t2 = time.time()
        logger.info(f"epoch time: {t2-t1:.5}, accumulation loss: {accu_loss:.6}")
        # 如果当前的轮数是测试轮数的倍数，就进行模型的测试
        if (epoch + 1) % args.test_epoch == 0:
            predict_demos(model, tokenizer)
            bleu, rl = auto_evaluate(model, test_loader, tokenizer)
            BLEU.append(bleu)
            ROUGE_L.append(rl)
            logger.info(f"BLEU: {round(bleu, 9)}, Rouge-L: {round(rl, 8)}")
            plt.plot(LOSS)
            plt.savefig(output_dir / f'{model.__class__.__name__}_loss_{args.epochs}_{loss_function.reduction}.png')
            plt.clf()
            plt.plot(BLEU)
            plt.savefig(output_dir / f'{model.__class__.__name__}_BLEU_{args.epochs}_{loss_function.reduction}.png')
            plt.clf()
            plt.plot(ROUGE_L)
            plt.savefig(output_dir / f'{model.__class__.__name__}_Rouge-L_{args.epochs}_{loss_function.reduction}.png')
        # 如果当前的轮数是保存轮数的倍数，就保存模型的参数和相关信息到文件中
        if (epoch + 1) % args.save_epoch == 0:
            filename = f"{model.__class__.__name__}_{epoch + 1}.bin"
            filename = output_dir / filename
            save_model(filename, model, args, tokenizer)

LOSS = []
BLEU = []
ROUGE_L =[]
if __name__ == "__main__":
    run()