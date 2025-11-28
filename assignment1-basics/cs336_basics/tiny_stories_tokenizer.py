from tokenizer import Tokenizer, train_bpe

max_vocab_size = 10000
input_path = "/Users/giovannibianco/Documents/NNDL/LLM_from_scratch/assignment1-basics/data/data/TinyStoriesV2-GPT4-train.txt"
special_tokens =['<|endoftext|>']

vocab, merges = train_bpe(input_path, max_vocab_size, special_tokens)

tokenizer_class = Tokenizer(vocab, merges, special_tokens)
ids = tokenizer_class.encode ('hello how are you')
print(ids)
