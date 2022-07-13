# [ML] Predicting Anime Popularity

## Data

Last dataset used uploaded to the drive with the `popularity` column.

## Usage

At the moment usage flow is to open a python console and run the command below.

### Train model

```python
from main import *
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
```

### From state file

```python
from main import *
ann = AnimeNeuralNetwork.from_filename(
    data_file_path='dataset/dataset.csv',
    state_file_path='states/score.70.99.state.pt',
)
```

### Test
With the neural network trained or loaded as `ann`
```python
anime = Anime.from_array(ann.data.values[0])
ann.test_anime(anime) # _[0] * 10 -> mal score
```