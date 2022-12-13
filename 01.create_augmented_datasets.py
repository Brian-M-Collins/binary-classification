# %%
import awswrangler as wr
import pandas as pd
import nlpaug.augmenter.word as nlpaw

from sklearn.model_selection import train_test_split

from notebooks.brian.generic_review_detection.src.data_utils import (
    undersample_majority,
    get_relevant_words,
    augment_text,
)

# %%
data = wr.s3.read_csv(
    "s3://*.csv"
)
data["label"] = data["label"].apply(lambda x: (1 if x is True else 0))

# %%
train_text, test_text, train_labels, test_labels = train_test_split(
    data["text"], data["label"], test_size=0.2, stratify=data["label"]
)

train = pd.concat(
    [
        pd.DataFrame(train_text).reset_index(drop=True),
        pd.DataFrame(train_labels).reset_index(drop=True),
    ],
    axis=1,
)

# %%
wr.s3.to_csv(train, path="s3://*.csv")

# %%
unbalanced_df = undersample_majority(train, 0.30)

# %%
wr.s3.to_csv(unbalanced_df, path="s3://*.csv")


# %%
unbalanced_df["text"] = unbalanced_df["text"].apply(lambda text: get_relevant_words(text, 128))

# %%
aug10p = nlpaw.ContextualWordEmbsAug(
    model_path="bert-base-uncased", aug_min=1, aug_p=0.1, action="substitute"
)  #

# %%
balanced_df = augment_text(unbalanced_df, aug10p, 8, 3)

# %%
test_df = pd.concat([pd.DataFrame(test_text), pd.DataFrame(test_labels)], axis=1)

# %%
wr.s3.to_csv(balanced_df, path="s3://*.csv")
wr.s3.to_csv(test_df, path="s3://*.csv")
