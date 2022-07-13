import ast
import os
import random
from typing import Literal, cast

import numpy as np
import pandas as pd
import torch
from torch import nn, device, cuda, optim
from torch.utils.data import DataLoader

from models import Anime as _Anime, MediaTypeEnum, SourceEnum, AnimeSeasonEnum, RatingEnum

DEVICE = device('cuda' if cuda.is_available() else 'cpu')


def bitfield(num, fixed_length=None):
    return ([0 for _ in range(fixed_length - len(bin(num)) + 2)] if fixed_length else []) + \
           [1 if digit == '1' else 0 for digit in bin(num)[2:]]


class Anime(_Anime):
    @classmethod
    def from_array(cls, arr: np.array):
        args_order = ['id', 'title', 'main_picture', 'source', 'synopsis', 'genres', 'alternative_titles',
                      'start_season', 'num_list_users', 'popularity', 'mean', 'broadcast', 'num_favorites',
                      'related_anime', 'studios', 'num_scoring_users', 'media_type', 'rating',
                      'average_episode_duration', 'num_episodes', 'statistics']

        kwargs = {arg: arr[i] for i, arg in enumerate(args_order)}
        dict_args = ["main_picture", "genres", "alternative_titles", "start_season", "broadcast", "related_anime",
                     "studios", "statistics"]
        for da in dict_args:
            if isinstance(kwargs[da], float) and np.isnan(kwargs[da]):
                kwargs.pop(da)
            else:
                kwargs[da] = ast.literal_eval(kwargs[da])  # safely parses string to dict

        return cls(**kwargs)

    def to_features(self, max_members_rank, max_genre, min_title_length, max_title_length):
        media_type_nodes = len(bin(MediaTypeEnum.max())) - 2
        source_nodes = len(bin(SourceEnum.max())) - 2
        season_nodes = len(bin(AnimeSeasonEnum.max())) - 2
        rating_nodes = len(bin(RatingEnum.max())) - 2

        return {
            "title_length": (len(self.title) - min_title_length) / (max_title_length - min_title_length),
            "english_main_title": 1 if self.title == self.alternative_titles.en else 0,
            "has_english_title": 1 if self.alternative_titles.en else 0,
            "sequel": 1 if any([ra.relation_type == "prequel" for ra in self.related_anime]) else 0,
            "genres": np.array(
                [1 if g_id in [g.id for g in self.genres] else 0 for g_id in range(1, max_genre + 1)]),
            "media_type": np.array(bitfield(self.media_type.value, fixed_length=media_type_nodes)),
            "source": np.array(bitfield(self.source.value, fixed_length=source_nodes)),
            "season": np.array(bitfield(self.start_season.season.value, fixed_length=season_nodes)),
            "rating": np.array(bitfield(self.rating.value, fixed_length=rating_nodes)),
            "members_rank": (self.popularity - 1) / (max_members_rank - 1),
            "score": self.mean / 10,
        }

    def to_vector(self, max_members_rank, max_genre, min_title_length, max_title_length) -> np.ndarray:
        fts = self.to_features(max_members_rank, max_genre, min_title_length, max_title_length)
        vector = []
        for ft in fts.values():
            if isinstance(ft, (list, np.ndarray)):
                vector = [*vector, *ft]
            else:
                vector.append(ft)
        return np.array(vector)

    @classmethod
    def vector_to_features(cls, vec: np.ndarray) -> dict:
        raise NotImplemented


class AnimeNeuralNetwork(nn.Module):
    data: pd.DataFrame
    features_couple: list[tuple[np.ndarray, float | int]] = []
    OUTPUT: Literal["score", "members"] = "score"

    def __init__(self, data_file_path, hidden_sizes, output_size):
        self.data = pd.read_csv(data_file_path)

        self.max_members_rank = max(self.data.popularity)
        self.max_genre = max([ag['id'] for ags in self.data.genres for ag in ast.literal_eval(ags)])
        anime_titles_length = [len(at) for at in self.data.title]
        self.min_title_length, self.max_title_length = min(anime_titles_length), max(anime_titles_length)

        self.parse_features()

        self.input_size = self.features_couple[0][0].shape[0]
        self.hidden_sizes, self.output_size = hidden_sizes, output_size

        super().__init__()
        lrs_sizes = [self.input_size, *self.hidden_sizes, self.output_size]
        self.linear_relu_stack = nn.Sequential()
        for size_i in range(len(lrs_sizes) - 1):
            self.linear_relu_stack.append(nn.Linear(lrs_sizes[size_i], lrs_sizes[size_i + 1]))
            self.linear_relu_stack.append(nn.ReLU())

    def forward(self, x):
        return self.linear_relu_stack(x)

    def parse_features(self):
        for i, row in enumerate(self.data.values):
            anime = Anime.from_array(row)
            vec = anime.to_vector(self.max_members_rank, self.max_genre, self.min_title_length, self.max_title_length)
            self.features_couple.append((vec[:-2], vec[{"score": -1, "members": -2}[self.OUTPUT]]))

        return self.features_couple

    def run(self, train_perc, valid_perc, learning_rate, epochs):
        """
        Trains the model
        :param train_perc: Percentage of the dataset to set as training data
        :param valid_perc: Percentage of the training data to set as validation data
        :param learning_rate: Percentage to learn each iteration
        :param epochs: number of iterations
        :return:
        """
        BATCH_SIZE = 10
        STATES_DIR = './states'
        os.makedirs(STATES_DIR, exist_ok=True)

        random.shuffle(self.features_couple)

        train_data = self.features_couple[:int(self.data.shape[0] * train_perc)]
        valid_data = self.features_couple[len(train_data):len(train_data) + int(len(train_data) * valid_perc)]
        test_data = self.features_couple[len(train_data) + len(valid_data):]

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)  # type:ignore
        valid_loader = DataLoader(valid_data, batch_size=1)  # type:ignore
        test_loader = DataLoader(test_data, batch_size=1)  # type:ignore

        self.to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        MAX_PATIENCE = 15
        OUTPUT_TOLERANCE = 0.1

        patience_counter = 0
        min_loss = np.inf
        accuracy = {"training": [], "validation": []}
        loss = {"training": [], "validation": []}
        for epoch in range(epochs):
            self.train()
            local_loss = 0
            correct = 0
            total = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.reshape(-1, self.input_size).to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                optimizer.zero_grad()
                output = self(x_batch.float())
                # print(output, y_batch)

                running_loss = nn.functional.mse_loss(output, y_batch.float())
                running_loss.backward()
                optimizer.step()
                local_loss += running_loss.item()

                correct += len([True for out_e, y_e in zip(output, y_batch) if abs(out_e - y_e) < OUTPUT_TOLERANCE])
                # classification
                # correct += len([True for out_e, y_e in zip(output, y_batch) if torch.argmax(out_e) == y_e])
                total += len(output)

            loss["training"].append(local_loss / len(train_loader))
            accuracy["training"].append(correct / total)
            # print(f"Training accuracy: {round(accuracy['training'][-1] * 100, 2)}%")

            local_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                self.eval()
                for x_batch, y_batch in valid_loader:
                    x_batch = x_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    output = self(x_batch.float())

                    running_loss = nn.functional.mse_loss(output, y_batch.float())
                    local_loss += running_loss.item()

                    correct += len([True for out_e, y_e in zip(output, y_batch) if abs(out_e - y_e) < 0.1])
                    # classification
                    # correct += len([True for out_e, y_e in zip(output, y_batch) if torch.argmax(out_e) == y_e])
                    total += len(output)

                loss["validation"].append(local_loss / len(valid_loader))
                accuracy["validation"].append(correct / total)

            if np.mean(loss["validation"]) < min_loss:
                print(f"Epoch {epoch:03}: (validation) loss decreased")
                torch.save(self.state_dict(), os.path.join(STATES_DIR, self.to_filename()))
                min_loss = np.mean(loss["validation"])
                patience_counter = 0
            else:
                print(f"Epoch {epoch:03}: (validation) loss INCREASED")
                patience_counter += 1
                if patience_counter > MAX_PATIENCE:
                    print(f"STOP: Max patience reached ({MAX_PATIENCE})")
                    break

            print(f"\t(train) loss: {np.mean(loss['training']):.4f} | accuracy: {(accuracy['training'][-1] * 100):.3f}")
            print(f"\t(validation) "
                  f"loss: {min_loss:.4f} | "
                  f"accuracy: {(accuracy['validation'][-1] * 100):.3f}")

    def load_state(self, state_file_path: str):
        self.load_state_dict(torch.load(state_file_path))

    def test_anime(self, anime: Anime):
        vector = anime.to_vector(self.max_members_rank, self.max_genre, self.min_title_length, self.max_title_length)
        with torch.no_grad():
            self.eval()
            return self(torch.from_numpy(vector[:self.input_size]).to(DEVICE).float())

    def to_filename(self):
        """
        Generates the network state filename depending on the parameters
        format: {score|members}.{number_of_hidden_layers}_*{hidden_layer_sizes}.{input_size}.state.pt
        examples:
            * score.1_60.99.state.pt
            * score.2_50_30.96.state.pt
        """
        number_of_hidden_layers = (len(self.linear_relu_stack) // 2) - 1
        hidden_layer_sizes = [self.linear_relu_stack[(hi + 1) * 2].in_features for hi in range(number_of_hidden_layers)]

        return f"{self.OUTPUT}.{'_'.join(str(hls) for hls in hidden_layer_sizes)}.{self.input_size}.state.pt"

    @classmethod
    def from_filename(cls, data_file_path, state_file_path):
        args = os.path.basename(state_file_path).split('.')
        output = cast(Literal["score", "members"], args[0])
        hidden_sizes = [int(hs) for hs in args[1].split('_')]
        input_size = int(args[2])

        cls.OUTPUT = output
        model = cls(data_file_path, hidden_sizes, 1)
        model.load_state(state_file_path)
        return model


if __name__ == "__main__":
    ann = AnimeNeuralNetwork(
        './dataset/dataset.csv',
        hidden_sizes=(70,),
        output_size=1,
    )
    ann.run(
        train_perc=0.8,
        valid_perc=0.2,
        learning_rate=0.001,
        epochs=200
    )
