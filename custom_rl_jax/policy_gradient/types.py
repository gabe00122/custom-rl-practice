from typing_extensions import TypedDict

Metrics = TypedDict('Metrics', {
    'reward': float,
    'length': int,
    'state_value': float,
    'td_error': float
})
