import numpy as np

def integrated_gradients(
    inp, 
    target_label_index,
    predictions_and_gradients,
    baseline,
    steps=50):
    if baseline is None:
        baseline = 0*inp
    assert(baseline.shape == inp.shape)

    # Scale input and compute gradients.
    scaled_inputs = [baseline + (float(i)/steps)*(inp-baseline) for i in range(0, steps+1)]
    predictions, grads = predictions_and_gradients(scaled_inputs, target_label_index)  # shapes: <steps+1>, <steps+1, inp.shape>
    
    avg_grads = np.average(grads[:-1], axis=0)
    integrated_gradients = (inp-baseline)*avg_grads 
    return integrated_gradients, predictions
