# this is where I'll build a transformer based neural network for producing toy story like text
import torch
import torch.nn.functional as F

from models import GPT

device = "mps"

# look up table encoder decoder from letters in words to numbers
def make_encoder_decoder(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    letters = sorted(set("".join(words)))
    encoder = {letter: i for i, letter in enumerate(letters)}
    decoder = {i: letter for i, letter in enumerate(letters)}
    return encoder, decoder


# partition the dataset into sequences
def make_sequences(x: torch.Tensor, sequence_size: int) -> torch.Tensor:
    sequences = []
    for i in range(0, x.shape[0] - sequence_size):
        sequences.append(x[i : i + sequence_size])
    return torch.stack(sequences)


# partition the dataset into mini batches randomly and return the batches
def make_batches(x: torch.Tensor, batch_size):
    n = x.shape[0]
    indices = torch.randperm(n)
    x = x[indices]
    for i in range(0, n, batch_size):
        yield x[i : i + batch_size, :-1], x[i : i + batch_size, 1:]


# load toy story script as text into a list
text: list[str] = []
with open(f"shakespeare.txt") as f:
    text = list(f.read())

encoder, decoder = make_encoder_decoder(text)
sequence_length = 128
# create the dataset
all_letters = torch.tensor([encoder[letter] for letter in text], dtype=torch.long)
sequences = make_sequences(all_letters, sequence_length + 1).to(device)

num_heads = 6
embedding_dim = 156  # must be divisible by num_heads
vocab_size = len(encoder)

model = GPT(vocab_size).to("mps")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

losses = []

# training loop
max_iters = 1
batch_size = 64
num_batches = sequences.shape[0] // batch_size
model.train().to("mps")
for i in range(max_iters):
    for batch_num, (x, y) in enumerate(make_batches(sequences, batch_size)):
        logits = model(x)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        y = y.flatten()
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if batch_num % 100 == 0:
            print(
                f"epoch: {i}, batch: [{batch_num:>4d}/{num_batches:>4d}], loss: {loss.item():.4f}"
            )

        if batch_num % 1000 == 0:
            # save checkpoint of model (optimizer and model state dict)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                "checkpoint.pt",
            )

    print(f"epoch: {i}, loss: {loss.item()}")
