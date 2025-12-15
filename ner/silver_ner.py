
"""
SILVER NER dataset generations using Wikipedia API.
"""

def tokenize_bn(text):
  return text.split()


def bio_tag(tokens, entities):
  tags= ["0"] * len(tokens)

  for ent_text, ent_label in entities:
    ent_tokens = ent_text.split()

    for i in range(len(tokens) - len(ent_tokens) + 1):
      if tokens[i:i+len(ent_tokens)] == ent_tokens:
        tags[i] = f"B-{ent_label}"
        for j in range(1, i+len(ent_tokens)):
          tags[i+j] = f"I-{ent_label}"

  return list(zip(tokens, tags))

