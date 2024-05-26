from pytorch_forecasting import NBeats, NHiTS, TemporalFusionTransformer
from pytorch_forecasting.metrics import MASE


def get_tft_model(ds, hidden_size):
    '''returns temporal fusion transformer model'''
    model = TemporalFusionTransformer.from_dataset(
        ds,
        learning_rate=1e-4,
        hidden_size=hidden_size,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=hidden_size,
        loss=MASE(),
        log_interval=10,
        optimizer='Adam',
        reduce_on_plateau_patience=100)

    return model
