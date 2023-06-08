import numpy as np
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


if __name__ == "__main__":
    oracle = tf_bind_8_oracle()

    example = np.array([0,0,0,0,0,0,0,1]).reshape(1,-1)
    example4 = np.array([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,2],[0,0,0,0,0,0,0,3]]).reshape(-1,8)

    pred = oracle.predict(example)
    print(pred)

    pred2 = oracle.predict(example4)
    print(pred2)

