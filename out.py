import pandas as pd, numpy as np

test = pd.read_csv('data/test.csv')

def write_submission(models, label_cols)
    preds = np.zeros((len(test), len(label_cols)))
    for i, model in enumerate(models):
        preds[:,i] = model.predict_proba(test['comment_text'])[:,1]

    ids = pd.DataFrame({'id': test["id"]})
    out_df= pd.concat([ids, pd.DataFrame(preds, columns = label_cols)], axis=1)
    submission.to_csv('submission.csv', index=False)

