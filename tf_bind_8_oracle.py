import numpy as np
import torch
from preprocessing.data_download import DATA_FOLDER


class tf_bind_8_oracle():
    def __init__(self):
        self.x = np.load(DATA_FOLDER + 'tf_bind_8/SIX6_REF_R1/tf_bind_8-x.npy')
        self.y = np.load(DATA_FOLDER + 'tf_bind_8/SIX6_REF_R1/tf_bind_8-y.npy')

    def predict(self, in_arr):
        #batch_size = len(in_arr)

        preds = []
        for arr in in_arr:
            pos = np.where(np.all(self.x == arr, axis=1) == True)[0]
            pred = self.y[pos][0][0]
            preds.append(pred)

        return preds

class tf_bind_8_oracle_pt():
    def __init__(self):
        self.x = torch.from_numpy(np.load(DATA_FOLDER + 'tf_bind_8/SIX6_REF_R1/tf_bind_8-x.npy'))
        self.y = torch.from_numpy(np.load(DATA_FOLDER + 'tf_bind_8/SIX6_REF_R1/tf_bind_8-y.npy'))

    def predict(self, in_arr):
        #batch_size = len(in_arr)

        preds = []
        for arr in in_arr:
            pos = torch.where(torch.all(self.x == arr, dim=1) == True)[0]
            pred = self.y[pos][0][0]
            preds.append(pred)

        return preds


if __name__ == "__main__":
    oracle = tf_bind_8_oracle()
    oracle_pt = tf_bind_8_oracle_pt()

    example = np.array([0,0,0,0,0,0,0,1]).reshape(1,-1)
    example4 = np.array([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,2],[0,0,0,0,0,0,0,3]]).reshape(-1,8)

    example_tensor = torch.from_numpy(example)
    example_tensor4 = torch.from_numpy(example4)

    pred = oracle.predict(example)
    pred_pt = oracle_pt.predict(example_tensor)
    print('pred', pred)
    print('pred_pt', pred_pt)

    pred2 = oracle.predict(example4)
    pred2_pt = oracle_pt.predict(example_tensor4)
    print(pred2)
    print(pred2_pt)

