# We care about these columns only
features = df[['position_x', 'position_y', 'position_z']]

# Choose 12 first flights for training, the next 4 for validation, and the last 2 (17,18) take them as test set
feature_cols = features.columns

train_raw = df.loc[df["uid"].between(1, 12), feature_cols]
val_raw   = df.loc[df["uid"].between(13, 16), feature_cols]

# Transform data so that each feature has a mean of 0 and a standard deviation of 1.
# Fit ONLY on training set, apply into validation to avoid data leakage
scaler = StandardScaler()
train_data = scaler.fit_transform(train_raw)
val_data = scaler.transform(val_raw)

# This is like sliding window to collect data
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, input_length, forecast_length):
        """
        V : # of variables , 3 (x,y,z)
        T : how many samples of x,y,z (lines of our CSV file) there are in data we pass to the function
        data: numpy array (T, V)
        returns:
            X: (V, input_length)
            y: (V, forecast_length)
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.input_length = input_length
        self.forecast_length = forecast_length

    # Len indicate the number of values that you can place the window and slice it , so it won't capture nan values
    def __len__(self):
        return len(self.data) - self.input_length - self.forecast_length + 1

    # idx is increasing by 1 . For example if input_length = forecast_length = 2 and idx = 0. That means x = [0,1] and y = [2,3] where x, y both
    # contain xyz pairs
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_length]              # (L, V)
        y = self.data[idx + self.input_length :
                      idx + self.input_length + self.forecast_length] # (F, V)

        # Put V (number of how many channels , 3 in our case) as rows and L or F as columns. Thats the format the model accepts
        x = x.T  # (L,V) -> (V, L) L: input length (how many previous samples we see)
        y = y.T  # (F,V) -> (V, F) F: forecast length

        return x, y
    
input_length = 96
forecast_length = 96    

train_dataset = TimeSeriesDataset(
    train_data, input_length, forecast_length
)

val_dataset = TimeSeriesDataset(
    val_data, input_length, forecast_length
)

# We do not shuffle data, its a trajectory so the order of points matter.
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=False,
)


val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
)
