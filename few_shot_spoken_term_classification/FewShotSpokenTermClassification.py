from typing import Union, Tuple, Optional, Callable
   
from functools import partial

from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset

from AudioLoader.speech import SPEECHCOMMANDS_12C


NOISE_LABEL = 10
UNKNOWN_LABEL = 11


def few_shot_target_transform(unknown_words: list[str], word_to_idx: list[str], _0, _1, _2, _3, word: str):
    if word in unknown_words:
        return UNKNOWN_LABEL
    elif word == '_silence_':
        return NOISE_LABEL
    else:
        return word_to_idx[word]


class FewShotSpokenTermClassification(Dataset):
    def __init__(self, task: str, root: Union[str, Path], url: str, folder_in_archive: str, transform: Optional[Callable] = None, download: bool = False, subset: str = "training"):
        """Create a Few-Shot Spoken Term Classification Dataset for N+M-way K-shot classification as per the paper "An Investigation
        of Few-Shot Learning in Spoken Term Classification" by Yanbin Chen, Tom Ko, Lifeng Shang, Xiao Chen, Xin Jiang, and Qing Li: https://arxiv.org/abs/1812.10233.

        Args:
            task (str): Which task to use. Either "command" or "digit"
            root (Union[str, Path]): Path to the directory where the dataset is found or downloaded.
            url (str): The URL to download the dataset from, or the type of the dataset to dowload. Allowed type values are `"speech_commands_v0.01"` and `"speech_commands_v0.02"`
            folder_in_archive (str): The top-level directory of the dataset. (default: `"SpeechCommands"`)
            transform (Optional[Callable], optional): Transform to apply to datasets. Defaults to None.
            download (bool, optional): Whether to download the dataset. Defaults to False.
            subset (str, optional): Select a subset of the dataset `["training", "validation", "testing"]`.
                `"validation"` and "testing" are defined in `"validation_list.txt"` and `"testing_list.txt"`,
                respectively, and "training" is the rest. Details for the files `"validation_list.txt"` and
                `"testing_list.txt"` are explained in the README of the dataset and in the introduction of Section
                7 of the original paper and its reference 12. The original paper can be found `here
                <https://arxiv.org/pdf/1804.03209.pdf>`. Defaults to "training".
        """

        self.task = task

        unknown_words = ['bed',  'dog', 'happy', 'marvin',  'wow']

        if self.task == "command":
            train_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'cat', 'tree', 'house', 'bird', 'visual', 'backward', 'follow', 'forward','learn','sheila']

            test_words = [
                'yes',
                'no',
                'up',
                'down',
                'left',
                'right',
                'on',
                'off',
                'stop',
                'go',
            ]
        elif task == "digit":
            train_words = ['go',
                'left',
                'learn',
                'bird',
                'visual',
                'cat',
                'yes',
                'on',
                'up',
                'no',
                'down',
                'right',
                'off',
                'house',
                'backward',
                'forward',
                'sheila',
                'stop',
                'tree',
                'follow'
            ]

            test_words = [
                'zero',
                'one',
                'two',
                'three',
                'four',
                'five',
                'six',
                'seven',
                'eight',
                'nine'
            ]
        else:
            raise NotImplementedError(f"Task '{self.task}' not implemented")
        
        if subset == "training" or subset == "validation":
            train_idx_to_word = {idx: word for idx, word in enumerate(train_words)}

            # Make sure that the UNKNOWN and NOISE classes have the same index during training
            # as they do during testing
            word_in_place_of_unknown = train_idx_to_word[UNKNOWN_LABEL]
            train_idx_to_word[len(train_words)] = word_in_place_of_unknown
            del train_idx_to_word[UNKNOWN_LABEL]

            word_in_place_of_noise = train_idx_to_word[NOISE_LABEL]
            train_idx_to_word[len(train_words)+1] = word_in_place_of_noise
            del train_idx_to_word[NOISE_LABEL]

            train_word_to_idx = {word: idx for idx, word in train_idx_to_word.items()}
            train_val_target_transform = partial(few_shot_target_transform, unknown_words, train_word_to_idx)

            train_or_val_data = SPEECHCOMMANDS_12C(root, url, folder_in_archive, download=download, subset=subset, transform=transform, target_transform=train_val_target_transform)
            train_or_val_subset = [i for i in range(len(train_or_val_data._data)) if train_or_val_data._data[i][5] in unknown_words + ['_silence_'] + train_words]

            self.data = Subset(train_or_val_data, train_or_val_subset)
        elif subset == "testing":
            test_idx_to_word = {idx: word for idx, word in enumerate(test_words)}

            test_word_to_idx = {word: idx for idx, word in test_idx_to_word.items()}
            test_target_transform = partial(few_shot_target_transform, unknown_words, test_word_to_idx)

            test_data = SPEECHCOMMANDS_12C(root, url, folder_in_archive, download=download, subset='testing', transform=transform, target_transform=test_target_transform)

            # Remove the training keywords from the test set
            test_subset = [i for i in range(len(test_data._data)) if test_data._data[i][5] and str(test_data._data[i][3]).split('_')[0] not in train_words]

            self.data = Subset(test_data, test_subset)

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        return self.data[index]
    
    def __len__(self) -> int:
        return len(self.data)
