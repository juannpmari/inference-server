
class KVCache:
    """
     KV caching inside the inference engine so that repeated prefixes or multi-turn dialogues reuse key/value states and skip recomputing attention for prefix tokens
    """
    def __init__(self):
        pass

    def put(self, key: str, value: str):
        pass
    
    def get(self, key: str):
        pass
        