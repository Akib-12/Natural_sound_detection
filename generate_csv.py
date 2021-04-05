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
    info_list = sorted(info_list, key=lambda x: x["target"])
    filtered_list = []
    now = 0
    cnt = 0
    for row in info_list:
        if row["target"] == now and cnt < 10:
            filtered_list.append(row)
            cnt += 1
            if cnt == 10:
                now += 1
                cnt = 0

    pd.DataFrame(filtered_list).to_csv('dataset.csv')


# %%
get_filtered_csv()
