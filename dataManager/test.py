import pandas as pd

a=pd.DataFrame({
    "A":[1,2,3,1],
    "B":["a","b","c","d"]
})

for id, group in a.groupby(["A"]):
    for idx,row in group.iterrows():
        row["B"]="f"

print(a)

b=pd.DataFrame({
    "A":[1,2,3],
    "C":["a1","a2","a3"]
})

c=pd.DataFrame({
    "D":["B1","B2","B3"]
})

print(pd.merge(a,b,on='A'))

print(pd.concat([a,c],axis=0))
# print(pd.concat([a,c]),axis=1)