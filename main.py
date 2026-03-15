import config
import dataset
import negative_sampling
import model
import trainer
import torch 

# establishing the data-pipeline and training the model

tokens = dataset.load_text8(config.DATA_PATH)
word2idx, idx2word, corpus, word_freq = dataset.build_vocabulary(tokens, config.MIN_COUNT, config.VOCAB_SIZE)

# indexing our corpus only to a million tokens due to computational constraints. 
subsampled = dataset.subsample(corpus, word_freq, word2idx, config.SUBSAMPLE_THRESHOLD)[:1_000_000]
skip_gram_dataset = dataset.SkipGramDataset(subsampled, config.WINDOW_SIZE)
pairs = torch.utils.data.DataLoader(skip_gram_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)

# core training loop
noise = negative_sampling.build_noise_distribution(word_freq, word2idx)
skip_gram_model = model.SkipGram(config.VOCAB_SIZE, config.EMBED_DIM)
trainer.train(skip_gram_model, pairs, noise, word2idx, idx2word)
