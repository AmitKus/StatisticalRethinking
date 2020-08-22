import pickle


def load_model_trace(filename):
    with open(filename, 'rb') as buff:
        data = pickle.load(buff)

    basic_model, trace = data['model'], data['trace']
    return basic_model, trace


def save_model_trace(output_path: str, model, trace):
    """Pickles PyMC3 model and trace"""
    with open(output_path, "wb") as buff:
        pickle.dump({"model": model, "trace": trace}, buff)
