import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict], div_length: int):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    result_batch["pad_lengths"] = torch.tensor(
        [elem["audio"].shape[-1] for elem in dataset_items]
    )
    max_length = result_batch["pad_lengths"].max().item()
    more_pad_length = (
        0 if max_length % div_length == 0 else div_length - max_length % div_length
    )

    result_batch["audio"] = torch.concatenate(
        [
            pad_sequence(
                [elem["audio"].squeeze(0) for elem in dataset_items], batch_first=True
            ),
            torch.zeros(len(dataset_items), more_pad_length),
        ],
        dim=-1,
    ).unsqueeze(1)

    return result_batch
