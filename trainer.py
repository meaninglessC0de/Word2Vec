import torch.optim 
import torch
import config
import numpy as np
import json

def train(model, pairs, noise, word2idx, idx2word):
    count = 0 
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = model.to(device)
    print("Device detected is: ", device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = config.EPOCHS * len(pairs)
    # scheduler for learning rate decay
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimiser, 
        start_factor=1.0, 
        end_factor=0.0001, 
        total_iters=total_steps
    )


    for epoch in range(config.EPOCHS):
        print(f"epoch {epoch+1}/{config.EPOCHS}")
    
        for (center, context) in pairs:
            center    = center.to(device)
            context   = context.to(device)
            negatives = np.random.choice(len(noise), size=config.BATCH_SIZE * config.NUM_NEGATIVES, p=noise)
            negatives = torch.tensor(negatives, dtype=torch.long).reshape(config.BATCH_SIZE, config.NUM_NEGATIVES)
            negatives = negatives.to(device)
            optimiser.zero_grad()
            loss = model(center, context, negatives)
            loss.backward()
            optimiser.step()
            scheduler.step()
            count += 1 
            if count % config.LOG_EVERY == 0:
                print(loss.item())
            
        
        
    torch.save(model.input_embeddings.weight.data, "embeddings/embeddings.pt")
    print("embeddings saved")

    
    with open("embeddings/word2idx.json", "w") as f:
        json.dump(word2idx, f)

    with open("embeddings/idx2word.json", "w") as f:
        json.dump(idx2word, f)
