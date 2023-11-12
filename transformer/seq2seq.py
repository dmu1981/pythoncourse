import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import math 
import random
from tqdm import tqdm
import transformer
import sequenceset
import swipes as swp
import balancedaccuracy
import infinityiterator
import checkpointmanager

if __name__ == "__main__":
    writer = SummaryWriter()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device is {DEVICE}")

    SEQ_LEN = 100
    BATCH_SIZE = 256
    EMB_DIM = 60

    swipes = swp.Swipes("training-data_50_swipes.txt", max_lines=None, max_seq_len=SEQ_LEN, device=DEVICE) 
    seq2seq = sequenceset.Seq2SeqDataset(swipes, swipes.get_sos_token(), swipes.get_eos_token(), swipes.get_pad_token(), SEQ_LEN, 1)  
    dataloader = torch.utils.data.DataLoader(seq2seq, batch_size=BATCH_SIZE, shuffle=True)      

    DROPOUT = False

    net = transformer.Transformer(
      seq_len=SEQ_LEN, 
      n_tokens=swipes.get_n_tokens(), 
      pad_token=swipes.get_pad_token(), 
      emb_dim=EMB_DIM, 
      intermediate_dim=EMB_DIM*4, 
      n_layers=2, 
      n_heads=6, 
      dropout=False).to(DEVICE)

    optim = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    chkptManager = checkpointmanager.CheckpointManager("swipes_v9")

    try:
        chkptName = chkptManager.get_best_checkpoint()
        print(f"Starting with checkpoint {chkptName}")
        chkpt = torch.load(chkptName)
        net.load_state_dict(chkpt["model"])
        optim.load_state_dict(chkpt["optim"])
    except:
        print("Could not load checkpoint, starting from scratch!")

    bar = tqdm(range(1,1000000))
    total_loss = 0
    cnt = 0
    nrm = 0

    balancedAccuracy = balancedaccuracy.BalancedAccuracy([swipes.get_pad_token(), swipes.get_sos_token()])
    dataiterator = infinityiterator.InfinityIterator(dataloader)

    for iter in bar:  
      inp, tgt, labels, sentence_weights = dataiterator.__next__()
      labels = labels.view(-1)

      optim.zero_grad()
      x = net(inp, tgt)
      loss = criterion(x, labels)
      w = torch.Tensor([swipes.token_weights[token.item()] for token in labels]).to(DEVICE)
      loss = torch.sum(loss * w * sentence_weights) / torch.sum(w * sentence_weights)

      loss.backward()

      balancedAccuracy.update(torch.argmax(x, dim=1).tolist(), labels.tolist())

      total_loss += loss.item()
      cnt += tgt.shape[0]
      nrm += torch.nn.utils.clip_grad_norm_(net.parameters(), 1000)
      optim.step()

      if iter % 100 == 0:
        net.eval()
        src = swipes.sequence_to_indices("HGFDSASDFGHJKLO")
        src = torch.concat((src, torch.ones(seq2seq.PAD_len - src.shape[0]).to(DEVICE) * seq2seq.PAD)).type(torch.long)
        res = net.seq2seq(src, swipes.get_sos_token(), swipes.get_eos_token())
        res = ''.join([swipes.token_to_char(token.item()) for token in res])
        print(res)
        net.train()

        bacc_dct, bacc, acc = balancedAccuracy.get()
        bacc_dct = { swipes.token_to_char(token): bacc_dct[token] for token in bacc_dct.keys() }
        balancedAccuracy.reset()

        bar.set_description(f"loss={total_loss / cnt * 1000.0:.3f}, acc={acc * 100:.3f}%, bacc={100*bacc:.3f}%, norm={tgt.shape[0] * nrm / cnt:.3f}")

        writer.add_scalar("loss", total_loss / cnt * 1000.0, iter)
        writer.add_scalar("accuracy", acc * 100, iter)
        writer.add_scalar("gradient_norm", tgt.shape[0] * nrm / cnt, iter)
        writer.add_scalar("balanced_accuracy", bacc * 100, iter)

        writer.add_scalars("token_accuracy", bacc_dct, iter)      

        total_loss = 0
        cnt, nrm = 0, 0

        try:
          chkpt = torch.save({
            "model": net.state_dict(),
            "optim": optim.state_dict(),
          }, chkptManager.new_checkpoint_file(bacc, clean_as_well=True))
        except:
          print("Could not save model!")
