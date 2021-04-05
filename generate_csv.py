# %%
import pandas as pd

# %%
df = pd.read_csv('esc50.csv')
print(df)


# %%
def get_filtered_csv():
    info_list = []
    for i in df.itertuples():
        info_list.append({
            "filename": i.filename,
            "target": i.target,
            "category": i.category
        })
    pd.DataFrame(info_list).to_csv('dataset_all.csv')


# %%
get_filtered_csv()
