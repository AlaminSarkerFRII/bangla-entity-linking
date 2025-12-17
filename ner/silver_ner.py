def tokenize_bn(text):
    """
    Simple whitespace tokenizer for Bangla
    (Can be replaced with Indic tokenizer later)
    """
    return text.strip().split()


def bio_tag(tokens, mention, label="ENT"):
    """
    Convert a single entity mention into BIO tags
    """
    tags = ["O"] * len(tokens)

    mention_tokens = mention.split()
    m_len = len(mention_tokens)

    for i in range(len(tokens) - m_len + 1):
        if tokens[i:i + m_len] == mention_tokens:
            tags[i] = f"B-{label}"
            for j in range(1, m_len):
                tags[i + j] = f"I-{label}"
            return tags

    return None
